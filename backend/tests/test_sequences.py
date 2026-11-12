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
