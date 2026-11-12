from app.services.spatial import generate_spatial


def frame(t,x=.1,state='static',missing=False):
    return dict(time=t,camera_state=state,items=[] if missing else [dict(id='person_1',class_name='person',box=[x,.2,x+.1,.7]),dict(id='object_1',class_name='cardboard box',box=[.6,.4,.7,.6])])


def test_temporal_proximity_and_direction():
    rows=generate_spatial([frame(i*.2,.1+i*.04) for i in range(11)])
    assert any(e['behavior']=='moves toward' for e in rows)
    assert all(e['review_status']=='needs_review' and 'unconfirmed' in e['description'] for e in rows)


def test_region_transition():
    region=dict(id='region_table',label='Table',kind='table',box=[.45,.1,.9,.9])
    rows=generate_spatial([frame(i*.2,.1 if i<7 else .6) for i in range(15)],[region])
    assert any(e['behavior']=='enters region' for e in rows)
    assert not any('placement' in e['behavior'] for e in rows)


def test_camera_motion_suppresses_region_and_direction():
    region=dict(id='region_table',box=[.4,.1,.9,.9])
    rows=generate_spatial([frame(i*.2,.1+i*.04,'moving') for i in range(11)],[region])
    assert not any(e['behavior'] in {'moves toward','moves away','enters region','exits region'} or e['details']['region_id'] for e in rows)


def test_missing_detections_and_legacy_camera_do_not_bridge():
    assert generate_spatial([frame(0),frame(.2),frame(.4,missing=True),frame(.6),frame(.8)])==[]
    rows=generate_spatial([dict(time=i*.2,items=frame(i*.2,.1+i*.04)['items']) for i in range(11)])
    assert not any(e['behavior'].startswith('moves ') for e in rows)


def test_cut_never_creates_region_transition():
    region=dict(id='table',box=[.45,.1,.9,.9])
    frames=[frame(i*.2,.1 if i<7 else .6) for i in range(15)]
    frames[7]['camera_state']='cut'
    assert not any(e['behavior']=='enters region' for e in generate_spatial(frames,[region]))


def test_overlap_does_not_claim_surface_placement():
    from app.services.spatial import surface_suggestions
    frames=[frame(i*.2,.55) for i in range(10)]
    regions=[dict(id='table',kind='table',box=[.4,.3,.9,.8])]
    assert any(e['behavior']=='overlapping' for e in generate_spatial(frames,regions))
    assert surface_suggestions([],frames,regions)==[]
    action=dict(id='place',behavior='placing / putting down',start_time=.5,end_time=1.4,participants=['person_1'],objects=[dict(id='object_1',class_name='cardboard box')])
    rows=surface_suggestions([action],frames,regions)
    assert len(rows)==1 and rows[0]['behavior']=='possible on-surface placement'
    assert rows[0]['details']['action_ids']==['place']


def test_region_entry_does_not_bridge_missing_object_observations():
    region=dict(id='table',box=[.45,.1,.9,.9])
    frames=[frame(i*.2,.1 if i<7 else .6,missing=i==6) for i in range(15)]
    assert not any(e['behavior']=='enters region' for e in generate_spatial(frames,[region]))
