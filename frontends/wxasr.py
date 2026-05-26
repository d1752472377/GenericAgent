"""WeChat voice ASR: silk → wav → text.
Primary: Groq Whisper (whisper-large-v3-turbo, ~2s).
Fallback: codex-asr local @ http://127.0.0.1:8788 (Bearer 123456).
不传 language 让模型自动检测 (硬编码 zh 会强制乱译)。
"""
import os, sys, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mykey

GROQ_URL = 'https://api.groq.com/openai/v1/audio/transcriptions'
GROQ_MODEL_DEFAULT = 'whisper-large-v3-turbo'
CODEX_URL = 'http://127.0.0.1:8788/v1/audio/transcriptions'
CODEX_KEY = '123456'  # 与启动脚本 --api-key 对齐


def _groq_cfg():
    return getattr(mykey, 'groq_config', None) or {}


def _post_whisper(url, key, wav_path, model, timeout):
    with open(wav_path, 'rb') as f:
        r = requests.post(
            url,
            headers={'Authorization': f'Bearer {key}'},
            files={'file': (os.path.basename(wav_path), f, 'audio/wav')},
            data={'model': model},
            timeout=timeout,
        )
    r.raise_for_status()
    return (r.json().get('text') or '').strip()


def transcribe_wav(wav_path):
    """Groq → 失败降级 codex-asr。返回文本或抛 RuntimeError。"""
    errs = []
    cfg = _groq_cfg()
    key = cfg.get('apikey')
    if key:
        try:
            model = cfg.get('asr_model', GROQ_MODEL_DEFAULT)
            return _post_whisper(GROQ_URL, key, wav_path, model, timeout=30)
        except Exception as e:
            errs.append(f'groq: {type(e).__name__}: {e}')
            print(f'[ASR] Groq失败, 降级codex-asr: {e}', file=sys.__stdout__)
    try:
        return _post_whisper(CODEX_URL, CODEX_KEY, wav_path, 'whisper-1', timeout=60)
    except Exception as e:
        errs.append(f'codex-asr: {type(e).__name__}: {e}')
    raise RuntimeError('ASR all backends failed: ' + ' | '.join(errs))


def silk_to_text(silk_path, *, cleanup=True):
    """微信 .silk → 解码为 wav → ASR 文本。
    成功后清理 silk+wav；失败保留 silk 便于排查，wav 仍清理。
    """
    import pilk
    base, ext = os.path.splitext(silk_path)
    wav_path = base + '.wav'
    pilk.silk_to_wav(silk_path, wav_path, rate=24000)
    ok = False
    try:
        text = transcribe_wav(wav_path)
        ok = True
        return text
    finally:
        # wav 总是清理；silk 只在成功时清理
        try: os.remove(wav_path)
        except OSError: pass
        if cleanup and ok:
            try: os.remove(silk_path)
            except OSError: pass


if __name__ == '__main__':
    # smoke test
    import sys as _s
    if len(_s.argv) > 1:
        print(silk_to_text(_s.argv[1], cleanup=False))
