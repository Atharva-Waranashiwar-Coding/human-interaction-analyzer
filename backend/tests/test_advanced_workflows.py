"""Small deterministic tests; no real videos or model weights are committed."""
import json
import pytest

from app.services.complex_activities import complex_candidates
from test_sequences import complete
from validate_dataset import validate_manifest
from temporal_experiment import load_rows
from app.services.error_analysis import error_report
from app.schemas import EvaluationRequest
from app.services.evaluation import evaluate


def clip(clip_id, split, group=None):
    return dict(clip_id=clip_id, scenario="static", duration=5,
                video_path=f"videos/{clip_id}.mp4",
                annotations_path=f"labels/{clip_id}.json",
                reviewer="independent-reviewer", split=split,
                source_group=group or clip_id, ground_truth_complete=True)


def test_dataset_manifest_validates_and_blocks_group_leakage():
    result=validate_manifest({"clips":[clip("a","train"),clip("b","validation"),clip("c","test")]})
    assert result["clips"]==3
    with pytest.raises(ValueError,match="leaks across splits"):
        validate_manifest({"clips":[clip("a","train","shared"),clip("b","test","shared")]})
    with pytest.raises(ValueError,match="exhaustive review"):
        validate_manifest({"clips":[dict(clip("a","train"),ground_truth_complete=False)]})
    with pytest.raises(ValueError,match="Unsafe"):
        validate_manifest({"clips":[dict(clip("a","train"),video_path="../../private.mp4")]})


def activity(action_list, source, destination, start):
    # Use complete action references, without synthetic spatial inferences.
    return dict(
        id=f"activity-{start}",behavior="object_relocation",annotation_kind="activity",
        start_time=start,end_time=start+2.5,
        participants=["p"],objects=action_list[0]["objects"],
        details=dict(sequence_status="complete",source_region=source,
                     destination_region=destination,
                     action_ids=[row["id"] for row in action_list]),
    )


def test_complex_removal_and_distribution_require_distinct_completed_tracks():
    regions=[dict(id="bin",kind="container"),dict(id="left",kind="table"),
             dict(id="right",kind="table")]
    first=activity(complete(o="o"),"bin","left",0)
    second=activity(complete(o="b",offset=4),"bin","right",4)
    rows=complex_candidates([],regions=regions,activity_rows=[first,second])
    assert {r["behavior"] for r in rows}=={
        "possible repeated container-region removal","possible multi-region object distribution"}
    assert all(r["review_status"]=="needs_review" and r["confidence"] is None for r in rows)
    assert all(len(r["details"]["action_ids"])==6 for r in rows)
    same_object=activity(complete(offset=4),"bin","right",4)
    assert complex_candidates([],regions=regions,activity_rows=[first,same_object])==[]


def test_complex_activities_reject_missing_region_and_partial_evidence():
    regions=[dict(id="bin",kind="container")]
    first=activity(complete(o="o"),"bin","left",0)
    second=activity(complete(o="b",offset=4),"bin","right",4)
    second["details"]["sequence_status"]="partial"
    assert complex_candidates([],regions=regions,activity_rows=[first,second])==[]
    second["details"]["sequence_status"]="complete"
    second["details"].pop("source_region")
    assert complex_candidates([],regions=regions,activity_rows=[first,second])==[]


def test_temporal_features_validate_splits_and_finite_values(tmp_path):
    rows=[dict(features=[[float(i),float(i+1)] for i in range(3)],
               label=label,split=split,group=group)
          for split,group in [("train","a"),("train","b"),
                              ("validation","c"),("test","d")]
          for label in ("walk","reach")]
    path=tmp_path/"features.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    loaded,shape,labels=load_rows(path)
    assert len(loaded)==8 and shape==(3,2) and labels==["reach","walk"]
    rows[0]["group"]="c"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError,match="leaks across splits"):
        load_rows(path)


def test_error_report_breaks_down_false_positive_and_missed_truth():
    original=dict(id="predict",behavior="carrying",start_time=1,end_time=2,
                  participants=["wrong"],objects=[],review_status="reviewed")
    pred=dict(original,id="review",original_suggestion=original)
    truth=dict(original,id="truth",participants=["p"],evaluation_role="ground_truth")
    evaluation=evaluate([pred,truth],EvaluationRequest(
        covered_behaviors=["carrying"],end_time=5,
        complete_ground_truth=True,scenario="occlusion"))
    report=error_report([evaluation])
    assert report["by_scenario"]["occlusion"]["false_positives"]==1
    assert report["by_behavior"]["carrying"]["false_negatives"]==1
    assert len(report["error_examples"])==2


def test_pose_feature_builder_normalizes_and_masks_missing_anchors():
    from build_pose_features import convert
    frame=[[float(i),float(i+3),.95] for i in range(17)]
    sample=dict(label="reach",split="train",group="clip_A",
                poses=[frame,frame,[[0.,0.,.1] for _ in range(17)]])
    output=convert(sample)
    assert output["label"]=="reach"
    assert len(output["features"])==3
    assert len(output["features"][0])==51
    assert output["features"][-1]==[0.0]*51
    assert output["features"][0][11*3+2]==1.0


def test_scoped_regression_comparison_rejects_different_evaluation_scopes():
    from app.services.error_analysis import compare_reports
    from evaluate_fixture import summarize
    original=dict(id="prediction",behavior="walking",start_time=0,end_time=1,
                  participants=["p"],objects=[])
    predicted=dict(original,id="review",review_status="reviewed",
                   original_suggestion=original)
    truth=dict(original,id="truth",review_status="reviewed",
               evaluation_role="ground_truth")
    req=dict(covered_behaviors=["walking"],end_time=3,
             complete_ground_truth=True,scenario="static")
    baseline=summarize([dict(request=req,annotations=[predicted,truth])])
    current=summarize([dict(request=req,annotations=[predicted,truth])])
    comparison=compare_reports(current,baseline)
    assert comparison["summary_delta"]["f1"]==0
    changed=summarize([dict(request=dict(req,scenario="moving"),
                            annotations=[predicted,truth])])
    with pytest.raises(ValueError,match="different evaluation scopes"):
        compare_reports(changed,baseline)
