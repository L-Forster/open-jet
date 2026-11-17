"""Real GGUF header fixtures exercise build and runtime selection without tensor data."""
import asyncio
import json
import re
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


@pytest.fixture(autouse=True)
def rtx_3090():
    with mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '8.6')):
        yield


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


def test_k_quant_files_select_the_engine(model):
    path = model()
    assert ls._megakernel_variant(path) == 'native'
    assert ls._megakernel_library(path) == 'libqwen38mk.so'
    # a plain llama-quantize Q4_K_M, and the UD-Q4_K_XL roles
    for overrides in ({'blk.0.ssm_out.weight': 12, 'blk.0.ssm_alpha.weight': 12, 'blk.0.ssm_beta.weight': 12,
                       'blk.64.nextn.eh_proj.weight': 12},
                      {'blk.0.ssm_out.weight': 8, 'blk.0.ssm_alpha.weight': 8, 'blk.64.nextn.eh_proj.weight': 14},
                      {'blk.0.ffn_gate.weight': 23}, {'blk.0.ffn_down.weight': 11}, {'blk.0.ffn_gate.weight': 21},
                      {'blk.0.ffn_gate.weight': 20}):
        assert ls._megakernel_variant(model(**overrides)) == 'native'


@pytest.mark.parametrize('overrides', [
    # the 1- and 2-bit types and IQ3_XXS (UD-Q3_K_XL, GSQ-RCO) and BF16: not in megakernel/gguf.h
    {'blk.0.attn_q.weight': 10}, {'blk.0.ffn_up.weight': 16}, {'blk.0.ffn_up.weight': 17},
    {'blk.0.ffn_gate.weight': 18}, {'blk.0.ffn_up.weight': 19}, {'blk.0.ffn_down.weight': 22},
    {'blk.0.ffn_gate.weight': 29},
    {'token_embd.weight': 22, 'output.weight': 12, 'blk.0.ssm_alpha.weight': 30, 'blk.0.ssm_beta.weight': 30},
])
def test_types_only_the_previous_engine_reads_select_its_mixed_build(model, tmp_path, overrides):
    path = model(**overrides)
    assert ls._megakernel_variant(path) == 'mixed'
    assert ls._megakernel_library(path) == 'libqwen38mk_mixed.so'
    with mock.patch.object(ls, '_megakernel_root', return_value=tmp_path):
        assert ls._megakernel_engine_root(path) == tmp_path / 'legacy'
        assert ls._megakernel_engine_root(model()) == tmp_path


@pytest.mark.parametrize('overrides', [
    {'blk.0.ffn_gate.weight': 24},  # I8: not a weight type either engine reads
    {'blk.0.ffn_gate.weight': 3},
])
def test_unsupported_types_are_rejected_before_building(model, overrides):
    assert ls._megakernel_variant(model(**overrides)) is None


def ud(path, **overrides):
    types = {'token_embd.weight': 12, 'output.weight': 14, 'blk.0.ssm_out.weight': 8, 'blk.0.ssm_alpha.weight': 8,
             'blk.64.nextn.eh_proj.weight': 14, 'blk.0.ffn_gate.weight': 23}
    types.update(overrides)
    return gguf(path, types)


def test_the_gpu_picks_between_the_engines(model, tmp_path):
    q4_k_m, xl = model(), ud(tmp_path / 'Qwen3.8-27B-UD-Q4_K_XL.gguf')
    q3 = ud(tmp_path / 'Qwen3.8-27B-UD-Q3_K_XL.gguf', **{'blk.0.ssm_out.weight': 13, 'blk.0.ffn_gate.weight': 11})
    # a 24 GB Ampere or newer GPU: the current engine, except for UD-Q3_K_XL
    assert [ls._megakernel_variant(path) for path in (q4_k_m, xl, q3)] == ['native', 'native', 'mixed']
    # no 24 GB GPU (a 16 GB card): the previous engine's mixed build takes the UD files
    with mock.patch.object(ls, '_megakernel_gpu_info', side_effect=lambda need=24000: ('0', '8.9') if need < 20000 else None):
        assert ls._megakernel_variant(xl) == 'mixed'
        assert ls._megakernel_variant(q3) == 'mixed'
        assert ls._megakernel_min_vram_mib(q3) < ls._MEGAKERNEL_MIN_VRAM_MIB
    with mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '7.5')):
        assert [ls._megakernel_variant(path) for path in (q4_k_m, xl)] == ['legacy', 'mixed']
        assert ls._megakernel_library(q4_k_m) == 'libqwen38mk.so'


def test_turing_serves_only_the_previous_native_engine(model, tmp_path):
    native, mixed = model(), ud(tmp_path / 'Qwen3.8-27B-UD-Q4_K_XL.gguf')
    with mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '7.5')), \
            mock.patch.object(ls, '_megakernel_dir', return_value=Path('/mk')), \
            mock.patch.object(ls.importlib.util, 'find_spec', return_value=object()):
        assert ls.model_runtime_label(native) == ls.MEGAKERNEL_TURING_LABEL
        assert ls.model_runtime_label(mixed) == 'llama.cpp'
        assert ls.model_runtime_label('/missing/Qwen3.8-27B-UD-Q4_K_XL.gguf') == 'llama.cpp'
        assert ls.megakernel_unavailable_reason(native) is None
        assert 'Turing' in ls.megakernel_unavailable_reason(mixed)
        assert ls.megakernel_long_context_reason(native) is not None
        log = mock.Mock()
        with mock.patch.object(mb.sys, 'platform', 'linux'), \
                mock.patch.object(mb, '_megakernel_gpu_info', return_value=('0', '7.5')), \
                mock.patch.object(mb, '_run_exec', new_callable=mock.AsyncMock) as run:
            asyncio.run(mb.ensure_megakernel({'llama_model': mixed}, log=log, set_status=mock.Mock(),
                                             clear_status=mock.Mock()))
        run.assert_not_called()
        assert 'Turing' in log.write.call_args.args[0]


def test_setup_builds_the_previous_engine_in_its_own_directory(tmp_path, model):
    root = tmp_path / 'build'
    legacy = root / 'legacy'
    (legacy / 'server').mkdir(parents=True)
    for name in ('Makefile', 'draft_vocab_32k.bin', 'server/mk_server.py'):
        (legacy / name).touch()
    calls = []

    async def run(*args):
        calls.append(args)
        (Path(args[args.index('-C') + 1]) / args[-1]).touch()
        return 0, '', ''

    q3 = ud(tmp_path / 'Qwen3.8-27B-UD-Q3_K_XL.gguf', **{'blk.0.ssm_out.weight': 13, 'blk.0.ffn_gate.weight': 11})
    with mock.patch.object(ls, '_megakernel_root', return_value=root), \
         mock.patch.object(mb, '_megakernel_root', return_value=root), \
         mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '7.5')), \
         mock.patch.object(mb, '_megakernel_gpu_info', return_value=('0', '7.5')), \
         mock.patch.object(mb, '_nvcc_path', return_value='nvcc'), \
         mock.patch.object(mb, '_llama_cpp_ready', return_value=True), \
         mock.patch.object(mb, 'install_prebuilt_megakernel', new_callable=mock.AsyncMock, return_value=False), \
         mock.patch.object(mb, 'read_memory_snapshot', return_value=mock.Mock(total_mb=15955.0)), \
         mock.patch.object(mb.sys, 'platform', 'linux'), \
         mock.patch.object(mb, '_run_exec', side_effect=run):
        for _ in range(2):
            asyncio.run(mb.ensure_megakernel({'llama_model': model()}, log=mock.Mock(), set_status=mock.Mock(),
                                             clear_status=mock.Mock()))
        with mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '8.6')), \
                mock.patch.object(mb, '_megakernel_gpu_info', return_value=('0', '8.6')):
            for _ in range(2):
                asyncio.run(mb.ensure_megakernel({'llama_model': q3}, log=mock.Mock(), set_status=mock.Mock(),
                                                 clear_status=mock.Mock()))
            assert ls._megakernel_dir(q3) == legacy
        # the 22 GiB guard is the current engine's: the previous one compiled on this 16 GB machine
        assert [(args[args.index('-C') + 1], args[-1]) for args in calls] == [
            (str(legacy), 'libqwen38mk.so'), (str(legacy), 'libqwen38mk_mixed.so')]
        assert 'ARCH=sm_75' in calls[0] and 'ARCH=sm_86' in calls[1]
        assert json.loads((legacy / '.openjet-build.json').read_text()) == {'arch': 'sm_75'}
        assert json.loads((legacy / '.openjet-build-mixed.json').read_text()) == {'arch': 'sm_86'}
        assert ls._megakernel_dir(model()) == legacy


def test_the_previous_engine_starts_with_its_own_options(model, tmp_path):
    client = ls.LlamaServerClient(model=model())
    client._spawn_until_healthy = mock.AsyncMock(side_effect=RuntimeError('captured command'))
    with mock.patch.object(ls, '_megakernel_gpu_info', return_value=('0', '7.5')), \
            mock.patch.object(ls, '_megakernel_root', return_value=tmp_path):
        (tmp_path / 'legacy' / 'server').mkdir(parents=True)
        for name in ('libqwen38mk.so', 'draft_vocab_32k.bin', 'server/mk_server.py'):
            (tmp_path / 'legacy' / name).touch()
        with pytest.raises(RuntimeError, match='captured command'):
            asyncio.run(client._start_megakernel(env={'CUDA_VISIBLE_DEVICES': '0'}, ctx=4096))
    cmd = client._spawn_until_healthy.call_args.args[0]
    assert cmd[1] == str(tmp_path / 'legacy' / 'server' / 'mk_server.py')
    assert cmd[cmd.index('--lib') + 1] == str(tmp_path / 'legacy' / 'libqwen38mk.so')
    assert not {'--kv', '--exact-reuse', '--park-mib', '-d'} & set(cmd)
    assert client._spawn_until_healthy.call_count == 1
    asyncio.run(client._http.aclose())


def test_the_previous_engine_makefile_keeps_its_two_builds_apart():
    root = Path(__file__).resolve().parents[1] / 'megakernel' / 'legacy'
    native = subprocess.check_output(['make', '-nB', '-C', str(root), 'ARCH=sm_75', 'libqwen38mk.so'], text=True)
    mixed = subprocess.check_output(['make', '-nB', '-C', str(root), 'ARCH=sm_86', 'libqwen38mk_mixed.so'], text=True)
    assert '-c qwen38_mk.cu' in native and '-DQWEN_MK_SM75=1' in native and 'qwen38_mixed' not in native
    assert '-c qwen38_mixed.cu' in mixed and '-c qwen38_mk.cu' not in mixed
    for name in mb._sources('legacy') + mb._sources('mixed'):
        assert (root / name).is_file(), name


def test_the_accepted_types_are_the_engine_readers(model):
    header = (Path(__file__).resolve().parents[1] / 'megakernel' / 'gguf.h').read_text()
    line = next(line for line in header.splitlines() if 'F32 = 0' in line)
    assert {int(value) for value in re.findall(r'= (\d+)', line)} == set(ls._MEGAKERNEL_TYPES)


def test_only_the_measured_files_are_served(tmp_path):
    assert ls._megakernel_model_supported('Qwen3.8-27B-Q4_K_M.gguf')
    assert ls._megakernel_model_supported('Qwen3.8-27B-UD-Q4_K_XL.gguf')
    assert ls._megakernel_model_supported('Qwen3.8-27B-UD-Q3_K_XL.gguf')
    # a catalog file not downloaded yet: unsloth UD-Q3_K_XL fits a 16 GB card
    assert 14000 < ls._megakernel_min_vram_mib(str(tmp_path / 'Qwen3.8-27B-UD-Q3_K_XL.gguf')) < 16000
    assert not ls._megakernel_model_supported('Qwen3.8-27B-UD-IQ3_XXS.gguf')
    assert ls._megakernel_min_vram_mib(str(tmp_path / 'Qwen3.8-27B-Q4_K_M.gguf')) == ls._MEGAKERNEL_MIN_VRAM_MIB


def test_truncated_header_is_rejected(tmp_path):
    path = tmp_path / 'broken.gguf'
    path.write_bytes(b'GGUF\x03')
    assert ls._megakernel_variant(str(path)) is None


def test_setup_builds_the_engine_once(tmp_path, model):
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
         mock.patch.object(mb, 'install_prebuilt_megakernel', new_callable=mock.AsyncMock, return_value=False), \
         mock.patch.object(mb, 'read_memory_snapshot', return_value=mock.Mock(total_mb=65536.0)), \
         mock.patch.object(mb.sys, 'platform', 'linux'), \
         mock.patch.object(mb, '_run_exec', side_effect=run):
        for overrides in ({}, {'blk.0.ffn_gate.weight': 23}):
            asyncio.run(mb.ensure_megakernel({'llama_model': model(**overrides)},
                        log=mock.Mock(), set_status=mock.Mock(), clear_status=mock.Mock()))
        assert [args[-1] for args in calls] == ['libqwen38mk.so']
        assert all('-j1' in args for args in calls)
        assert json.loads((root / '.openjet-build.json').read_text()) == {'arch': 'sm_86'}
        assert ls._megakernel_dir(model()) == root


def test_make_builds_the_library_from_the_device_and_host_sources():
    root = Path(__file__).resolve().parents[1] / 'megakernel'
    plan = subprocess.check_output(['make', '-nB', '-C', str(root), 'ARCH=sm_86', 'libqwen38mk.so'], text=True)
    for source in ('qwen38_mk.cu', 'engine.cu', 'loader.cu', 'lanes.cu', 'capi.cpp'):
        assert source in plan
        assert source in mb._SOURCES


def test_runtime_passes_the_library(model, tmp_path):
    client = ls.LlamaServerClient(model=model())
    client._spawn_until_healthy = mock.AsyncMock(side_effect=RuntimeError('captured command'))
    with mock.patch.object(ls, '_megakernel_dir', return_value=tmp_path):
        with pytest.raises(RuntimeError, match='captured command'):
            asyncio.run(client._start_megakernel(env={'CUDA_VISIBLE_DEVICES': '0'}, ctx=4096))
    cmd = client._spawn_until_healthy.call_args.args[0]
    assert cmd[cmd.index('--lib') + 1] == str(tmp_path / 'libqwen38mk.so')
    asyncio.run(client._http.aclose())


def test_setup_does_not_guess_a_build_before_model_download(tmp_path):
    with mock.patch.object(mb, "_run_exec", new_callable=mock.AsyncMock) as run:
        asyncio.run(mb.ensure_megakernel(
            {"llama_model": str(tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf")},
            log=mock.Mock(), set_status=mock.Mock(), clear_status=mock.Mock()))
    run.assert_not_called()


def test_an_untested_quant_mix_is_labelled_semi_compatible(model):
    with mock.patch.object(ls, "_megakernel_gpu_info", return_value=("0", "8.6")):
        assert ls.model_runtime_label(model()) == ls.MEGAKERNEL_RUNTIME_LABEL
        assert ls.model_runtime_label(model(**{'blk.0.ssm_out.weight': 12})) == ls.MEGAKERNEL_SEMI_LABEL
        assert ls.model_runtime_label('/missing/Qwen3.8-27B-UD-Q4_K_XL.gguf') == ls.MEGAKERNEL_RUNTIME_LABEL


def test_setup_does_not_compile_without_the_ram_for_it(tmp_path, model):
    root = tmp_path / 'build'
    root.mkdir()
    (root / 'Makefile').touch()
    log = mock.Mock()
    with mock.patch.object(ls, '_megakernel_root', return_value=root), \
         mock.patch.object(mb, '_megakernel_root', return_value=root), \
         mock.patch.object(mb, '_megakernel_gpu_info', return_value=('0', '8.6')), \
         mock.patch.object(mb, '_nvcc_path', return_value='nvcc'), \
         mock.patch.object(mb, '_llama_cpp_ready', return_value=True), \
         mock.patch.object(mb, 'install_prebuilt_megakernel', new_callable=mock.AsyncMock, return_value=False), \
         mock.patch.object(mb, 'read_memory_snapshot', return_value=mock.Mock(total_mb=15955.0)), \
         mock.patch.object(mb.sys, 'platform', 'linux'), \
         mock.patch.object(mb, '_run_exec', new_callable=mock.AsyncMock) as run:
        asyncio.run(mb.ensure_megakernel({'llama_model': model()}, log=log, set_status=mock.Mock(),
                                         clear_status=mock.Mock()))
    run.assert_not_called()
    assert '22 GiB of RAM' in log.write.call_args.args[0]
