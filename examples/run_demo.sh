#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────
#  🦀 SCIENTIFIC TELEPHONE — FULL PIPELINE DEMO
#  Single command: audio → bridge → features → Q&A → phone
# ──────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   📞 SCIENTIFIC TELEPHONE — FULL PIPELINE DEMO     ║"
echo "║   audio → bridge → features → persona → phone out  ║"
echo "╚══════════════════════════════════════════════════════╝"

# 1. Check services
echo ""
echo "─── Step 1: Checking pipeline services ───"
if fuser 8765/tcp &>/dev/null; then
    echo "  ✅ OpenSMILE bridge: ws://0.0.0.0:8765"
else
    echo "  ❌ OpenSMILE bridge not running!"
    echo "     Run: sudo systemctl start fleet-opensmile"
    exit 1
fi

# 2. Check python + dependencies
echo ""
echo "─── Step 2: Checking dependencies ───"
python3.11 -c "import numpy; import websockets" 2>/dev/null && echo "  ✅ numpy + websockets" || echo "  ⚠ pip install numpy websockets"

# 3. Generate test speech audio
echo ""
echo "─── Step 3: Generating test speech audio ───"
python3.11 -c "
import numpy as np
sr = 16000
t = np.linspace(0, 3, sr * 3, False)
f0 = 180
audio = np.sin(2*np.pi*f0*t)*0.5
audio += np.sin(2*np.pi*f0*2*t)*0.25
audio += np.sin(2*np.pi*f0*3*t)*0.125
# Pitch glide for question intonation
for i, f in enumerate(np.linspace(f0, 240, sr)):
    if i < sr: audio[i] = 0.5*np.sin(2*np.pi*f*i/sr)
am = (1 + 0.5*np.sin(2*np.pi*4*t)) / 1.5
audio = audio * am
audio /= np.max(np.abs(audio))
audio.astype(np.float32).tofile('/tmp/test-speech.raw')
print('  ✅ Generated 3s speech audio → /tmp/test-speech.raw')
print('  ✅ F0: 180 Hz → 240 Hz glide')
"

# 4. Send to bridge and capture features
echo ""
echo "─── Step 4: Audio → Bridge → Features ───"
python3.11 -c "
import asyncio, websockets, json, numpy as np

async def test():
    audio = np.fromfile('/tmp/test-speech.raw', dtype=np.float32)
    async with websockets.connect('ws://localhost:8765') as ws:
        await ws.send(audio.tobytes())
        try:
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            f = json.loads(resp)['data']
            print(f'  ✅ Received {len(f)} feature dimensions:')
            for k in ['frame','loudness','f0_raw','spectral_centroid','alpha_ratio']:
                print(f'     {k:20s} = {f[k]}')
            with open('/tmp/last-features.json','w') as fp:
                json.dump(f, fp, indent=2)
        except asyncio.TimeoutError:
            print('  ❌ No features returned')
asyncio.run(test())
" 2>&1

# 5. Ensure synthetic Feynman persona exists
echo ""
echo "─── Step 5: Loading Feynman persona ───"
mkdir -p /home/ubuntu/.openclaw/workspace/persona-engine/memory
if [ ! -f /home/ubuntu/.openclaw/workspace/persona-engine/memory/feynman_demo.json ]; then
    python3.11 -c "
import json
persona = {
    'name': 'Richard Feynman',
    'cadence': {'pause_frequency': 0.15, 'pause_duration_mean': 0.35, 'rhythm_consistency': 0.7, 'mean_wpm': 187},
    'prosody': {'mean_f0': 145.0, 'f0_std': 28.0, 'f0_range': [80.0, 210.0], 'mean_energy': 0.45, 'energy_std': 0.15},
    'groove': {'bpm': 138, 'swing': 0.12, 'anticipation_ms': 180},
    'speaking_rate_wpm': 187,
    'turn_style': 'overlap_friendly'
}
with open('/home/ubuntu/.openclaw/workspace/persona-engine/memory/feynman_demo.json', 'w') as f:
    json.dump(persona, f, indent=2)
" 2>&1
    echo '  ✅ Created synthetic Feynman persona'
else
    echo '  ✅ Feynman persona exists'
fi

# 6. Run Scientific Telephone Q&A
echo ""
echo "─── Step 6: Scientific Telephone Q&A ───"
PYTHONPATH="/home/ubuntu/.openclaw/workspace/opensmile-bridge:/home/ubuntu/.openclaw/workspace/persona-engine" \
    python3.11 examples/scientific_telephone.py \
    --ask "What is the conservation theorem and how does it apply to avoidance dominance?" 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅ PIPELINE DEMO COMPLETE                         ║"
echo "║   Audio → Bridge → Features → Persona → Phone Out  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  View phone-filtered audio:"
echo "    ls output/*_phone.wav"
echo "  Listen:"
echo "    aplay output/*_phone.wav"
echo ""
