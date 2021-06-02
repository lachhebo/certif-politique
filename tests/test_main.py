def test_root_api_is_working(client):
    # given

    # when
    rv = client.get('/')

    # then
    assert rv.status_code == 200
