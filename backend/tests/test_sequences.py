from app.services.sequences import recognize_activities, ActivityDefinition, Step


def action(label,t,person='p',obj='o',duration=.5):
    return dict(id=f'{label}-{t}-{person}-{obj}',behavior=label,start_time=t,end_time=t+duration,participants=[person],objects=[dict(id=obj,class_name='cardboard box')])


def complete(p='p',o='o',offset=0):
    return [action(label,t+offset,p,o) for label,t in [('picking up / lifting',0),('carrying',1),('placing / putting down',2)]]


def test_complete_and_optional_steps_reference_actions():
    rows=recognize_activities(complete());assert len(rows)==1
    assert rows[0]['details']['sequence_status']=='complete'
    assert rows[0]['details']['action_ids']==[e['id'] for e in complete()]
    assert 'source_region' not in rows[0]['details']


def test_partial_missing_intermediate_and_ordering():
    assert recognize_activities(complete()[:1])[0]['details']['sequence_status']=='partial'
    row=recognize_activities([complete()[0],complete()[2]])[0]
    assert row['details']['sequence_status']=='partial' and len(row['details']['action_ids'])==2
    rows=recognize_activities([action('placing / putting down',0),action('carrying',1),action('picking up / lifting',2)])
    assert rows[0]['details']['sequence_status']=='partial'
    assert len(rows[0]['details']['action_ids'])==1


def test_concurrent_people_and_changed_object_ids():
    assert len(recognize_activities(complete()+complete('q','b')))==2
    rows=recognize_activities([complete()[0],action('carrying',1,obj='changed'),complete()[2]])
    assert rows[0]['details']['sequence_status']=='partial'


def test_interrupted_and_rejected():
    rows=recognize_activities([complete()[0],action('carrying',10)])
    assert rows[0]['details']['sequence_status']=='interrupted'
    rows=recognize_activities(complete(),[dict(time=.8,camera_state='cut',items=[])])
    assert rows[0]['details']['sequence_status']=='interrupted'
    assert recognize_activities([dict(complete()[0],review_status='rejected')])==[]


def test_repeated_and_custom_definition():
    rows=recognize_activities(complete()+complete(o='b',offset=4))
    assert any(e['behavior']=='repeated_pickup_placement' for e in rows)
    definition=ActivityDefinition('custom',(Step(('picking up / lifting',)),Step(('releasing',),optional=True)))
    assert recognize_activities(complete()[:1],definitions=[definition])[0]['details']['sequence_status']=='complete'


def test_repeated_destination_region_placement_requires_complete_distinct_objects():
    # Actual image-region lookup, no mocked spatial conclusions.
    regions=[dict(id='container_A',kind='container',box=[.5,.3,.9,.8])]
    frames=[dict(time=i/5,camera_state='static',items=[
        dict(id='p',class_name='person',box=[.1,.1,.2,.7]),
        dict(id='o',class_name='cardboard box',box=[.6,.4,.7,.6]),
        dict(id='b',class_name='cardboard box',box=[.7,.4,.8,.6]),
    ]) for i in range(36)]
    actions=complete()+complete(o='b',offset=4)
    rows=recognize_activities(actions,frames,regions)
    matching=[r for r in rows if r['behavior']=='repeated_destination_region_placement']
    assert len(matching)==1
    row=matching[0]
    assert row['details']['destination_region']=='container_A'
    assert row['details']['destination_region_kind']=='container'
    assert len(row['details']['action_ids'])==6
    assert {o['id'] for o in row['objects']}=={'o','b'}
    assert row['confidence'] is None and row['review_status']=='needs_review'
    assert 'not established' in row['description']
    assert not any(r['behavior']=='packing' for r in rows)


def test_region_grouping_never_claims_success_without_reliable_evidence():
    regions=[dict(id='container_A',kind='container',box=[.5,.3,.9,.8])]
    frames=[dict(time=i/5,camera_state='moving',items=[
        dict(id='p',class_name='person',box=[.1,.1,.2,.7]),
        dict(id='o',class_name='cardboard box',box=[.6,.4,.7,.6]),
        dict(id='b',class_name='cardboard box',box=[.7,.4,.8,.6]),
    ]) for i in range(36)]
    actions=complete()+complete(o='b',offset=4)
    assert not any(r['behavior']=='repeated_destination_region_placement'
                   for r in recognize_activities(actions,frames,regions))
    # Repeated movement of one object is not proof of several objects being packed.
    frames=[dict(f,camera_state='static') for f in frames]
    same_object=complete()+complete(offset=4)
    assert not any(r['behavior']=='repeated_destination_region_placement'
                   for r in recognize_activities(same_object,frames,regions))
