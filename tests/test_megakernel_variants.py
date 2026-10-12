"""Real GGUF header fixtures exercise build and runtime selection without tensor data."""
import asyncio
import json
import os
import struct
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from src import llama_server as ls
from src import megakernel_build as mb


def gguf(path, types):
    with path.open('wb') as f:
        f.write(b'GGUF' + struct.pack('<IQQ', 3, len(types), 0))
        for name, kind in types.items():
            name = name.encode()
            f.write(struct.pack('<Q', len(name)) + name)
            f.write(struct.pack('<IQIQ', 1, 256, kind, 0))
    return str(path)


@pytest.fixture
def model(tmp_path):
    def make(**overrides):
        types = {'token_embd.weight': 12, 'output.weight': 14,
                 'blk.0.ssm_out.weight': 13, 'blk.0.ssm_alpha.weight': 0,
                 'blk.0.ssm_beta.weight': 0, 'blk.64.nextn.eh_proj.weight': 8,
                 'blk.0.ffn_gate.weight': 12, 'blk.0.ffn_down.weight': 14}
        types.update(overrides)
        # Same filename for all tensor recipes: selection must come from the header.
        return gguf(tmp_path / 'Qwen3.8-27B-Q4_K_M.gguf', types)
    return make


def test_native_and_plain_q4_keep_the_original_engine(model):
    path = model()
    assert ls._megakernel_variant(path) == 'native'
    assert ls._megakernel_library(path) == 'libqwen38mk.so'
    path = model(**{'blk.0.ssm_out.weight': 12, 'blk.0.ssm_alpha.weight': 12,
                    'blk.0.ssm_beta.weight': 12, 'blk.64.nextn.eh_proj.weight': 12})
    assert ls._megakernel_variant(path) == 'native'


@pytest.mark.parametrize('overrides', [
    {'blk.0.ffn_gate.weight': 23}, {'blk.0.ffn_down.weight': 11},
    {'blk.0.ffn_gate.weight': 21}, {'blk.0.ffn_gate.weight': 20},
    # Q8_0 is an existing type, but not supported in this role by the native engine.
    {'blk.0.ssm_out.weight': 8, 'blk.0.ssm_alpha.weight': 8,
     'blk.64.nextn.eh_proj.weight': 14},
])
def test_extended_types_and_roles_select_mixed_engine(model, overrides):
    path = model(**overrides)
    assert ls._megakernel_variant(path) == 'mixed'
    assert ls._megakernel_library(path) == 'libqwen38mk_mixed.so'


@pytest.mark.parametrize('overrides', [
    {'token_embd.weight': 8}, {'output.weight': 12}, {'blk.0.ffn_gate.weight': 2},
])
def test_unsupported_types_are_rejected_before_building(model, overrides):
    assert ls._megakernel_variant(model(**overrides)) is None


def test_truncated_header_is_rejected(tmp_path):
    path = tmp_path / 'broken.gguf'
    path.write_bytes(b'GGUF\x03')
    assert ls._megakernel_variant(str(path)) is None


def test_setup_builds_only_selected_variant_and_keeps_both_cached(tmp_path, model):
    root = tmp_path / 'build'
    root.mkdir()
    (root / 'Makefile').touch()
    (root / 'draft_vocab_32k.bin').touch()
    (root / 'server').mkdir()
    (root / 'server/mk_server.py').touch()
    calls = []

    async def run(*args):
        calls.append(args)
        (root / args[-1]).touch()
        return 0, '', ''

    with mock.patch.object(ls, '_megakernel_root', return_value=root), \
         mock.patch.object(mb, '_megakernel_root', return_value=root), \
         mock.patch.object(mb, '_megakernel_gpu_info', return_value=('0', '8.6')), \
         mock.patch.object(mb, '_nvcc_path', return_value='nvcc'), \
         mock.patch.object(mb, '_llama_cpp_ready', return_value=True), \
         mock.patch.object(mb, '_run_exec', side_effect=run):
        for overrides in ({}, {'blk.0.ffn_gate.weight': 23}, {}, {'blk.0.ffn_gate.weight': 23}):
            asyncio.run(mb.ensure_megakernel({'llama_model': model(**overrides)},
                        log=mock.Mock(), set_status=mock.Mock(), clear_status=mock.Mock()))
        assert [args[-1] for args in calls] == ['libqwen38mk.so', 'libqwen38mk_mixed.so']
        assert all('-j1' in args for args in calls)
        for tag in ('.openjet-build.json', '.openjet-build-mixed.json'):
            assert json.loads((root / tag).read_text()) == {'arch': 'sm_86'}
        # Editing the optional engine does not invalidate the native build cache.
        changed = root / 'qwen38_mixed.cu'
        changed.touch()
        newer = (root / 'libqwen38mk_mixed.so').stat().st_mtime + 1
        os.utime(changed, (newer, newer))
        assert mb._build_is_current(root, 'sm_86', model())
        assert not mb._build_is_current(root, 'sm_86', model(**{'blk.0.ffn_gate.weight': 23}))
        # Only the selected binary satisfies runtime availability.
        path = model(**{'blk.0.ffn_gate.weight': 23})
        (root / 'libqwen38mk_mixed.so').unlink()
        assert ls._megakernel_dir(path) is None
        assert ls._megakernel_dir(model()) == root


def test_native_make_target_does_not_compile_mixed_source():
    root = Path(__file__).resolve().parents[1] / 'megakernel'
    native = subprocess.check_output(['make', '-nB', '-C', str(root), 'ARCH=sm_86',
                                      'libqwen38mk.so'], text=True)
    mixed = subprocess.check_output(['make', '-nB', '-C', str(root), 'ARCH=sm_86',
                                     'libqwen38mk_mixed.so'], text=True)
    assert '-c qwen38_mk.cu' in native
    assert 'qwen38_mixed' not in native
    assert '-c qwen38_mixed.cu' in mixed
    assert '-c qwen38_mk.cu' not in mixed


def test_runtime_passes_selected_library(model, tmp_path):
    path = model(**{'blk.0.ffn_gate.weight': 23})
    client = ls.LlamaServerClient(model=path)
    client._spawn_until_healthy = mock.AsyncMock(return_value='healthy')
    # Exercise command construction, then stop before starting a process or health polling.
    client._spawn_until_healthy.side_effect = RuntimeError('captured command')
    with mock.patch.object(ls, '_megakernel_dir', return_value=tmp_path):
        with pytest.raises(RuntimeError, match='captured command'):
            asyncio.run(client._start_megakernel(env={'CUDA_VISIBLE_DEVICES': '0'}, ctx=4096))
    args, kwargs = client._spawn_until_healthy.call_args
    cmd = kwargs.get('cmd', args[0] if args else [])
    assert cmd[cmd.index('--lib') + 1] == str(tmp_path / 'libqwen38mk_mixed.so')
    asyncio.run(client._http.aclose())


def test_setup_does_not_guess_a_build_before_model_download(tmp_path):
    with mock.patch.object(mb, "_run_exec", new_callable=mock.AsyncMock) as run:
        asyncio.run(mb.ensure_megakernel(
            {"llama_model": str(tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf")},
            log=mock.Mock(), set_status=mock.Mock(), clear_status=mock.Mock()))
    run.assert_not_called()


def test_mixed_runtime_label_does_not_promise_q4_performance(model):
    with mock.patch.object(ls, "_megakernel_gpu", return_value="0"):
        assert ls.model_runtime_label(model(**{'blk.0.ffn_gate.weight': 23})) == ls.MEGAKERNEL_MIXED_LABEL
        assert ls.model_runtime_label('/missing/Qwen3.8-27B-UD-Q4_K_XL.gguf') == ls.MEGAKERNEL_MIXED_LABEL
