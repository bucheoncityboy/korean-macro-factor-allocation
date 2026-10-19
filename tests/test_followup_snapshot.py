import pytest

from etf_regimes.followup_snapshot import pack_files, restore_files


def test_snapshot_is_deterministic_and_restores_exact_bytes(tmp_path):
    files = ['outputs/a.bin', 'outputs/b.txt']
    (tmp_path/'outputs').mkdir()
    payload = [bytes(range(256)) * 20, '원본 결과\n'.encode()]
    for name, value in zip(files, payload):
        (tmp_path/name).write_bytes(value)
    first = pack_files(tmp_path, files, chunk_bytes=127)
    second = pack_files(tmp_path, files, chunk_bytes=127)
    assert first == second
    for name in files:
        (tmp_path/name).unlink()
    result = restore_files(tmp_path, first, files)
    assert result['restored_files'] == 2
    for name, value in zip(files, payload):
        assert (tmp_path/name).read_bytes() == value


def test_snapshot_refuses_corrupt_shard(tmp_path):
    (tmp_path/'a').write_bytes(b'original')
    manifest = pack_files(tmp_path, ['a'])
    (tmp_path/manifest['parts'][0]['path']).write_bytes(b'corrupted')
    (tmp_path/'a').unlink()
    with pytest.raises(RuntimeError, match='fragment mismatch'):
        restore_files(tmp_path, manifest, ['a'])
    assert not (tmp_path/'a').exists()


def test_snapshot_never_overwrites_mismatched_existing_output(tmp_path):
    (tmp_path/'a').write_bytes(b'original')
    manifest = pack_files(tmp_path, ['a'])
    (tmp_path/'a').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='refusing overwrite'):
        restore_files(tmp_path, manifest, ['a'])
    assert (tmp_path/'a').read_bytes() == b'changed'
