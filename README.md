# Cloud Ops & Security Auditor (OpenEnv)
An automated RL environment designed to optimize cloud infrastructure costs and fix critical security leaks.

## 🚀 The Mission
Our agent acts as a Junior Cloud Architect. It must:
1. **Easy:** Identify and terminate idle servers (CPU < 5%) to save costs.
2. **Medium:** Detect world-exposed SSH (0.0.0.0) and restrict access.
3. **Hard:** Right-size instance tiers (Nano/Standard/Performance) to hit a target Cost-to-Performance ratio.

## 🛠️ Tech Stack
- **Environment:** Python, OpenEnv SDK, FastAPI.
- **Brain:** Gemini 1.5 Flash (via OpenAI-compatible API).
- **Deployment:** Hugging Face Spaces (Docker).

## 📊 Performance
- **Oracle Verified:** The environment is mathematically solvable with a 1.0 score.
- **Self-Correction:** The agent includes a retry loop to handle invalid LLM outputs.

## 🚀 Live Demo
- **Hugging Face Space:** https://huggingface.co/spaces/Vignan12/cloud-ops-env
- **GitHub Repository:** https://github.com/Vignan12/cloud-ops-env

## 🎯 How to Test
```bash
# Test against the deployed environment
python -m cloud_ops_env.inference https://vignan12-cloud-ops-env.hf.space

# Run locally
python -m uvicorn cloud_ops_env.server.app:app --host 0.0.0.0 --port 8000
```

## 📝 Environment Details
- **State Space:** Server configurations, CPU utilization, security status
- **Action Space:** terminate_server, fix_ssh_exposure, set_instance_tier, noop
- **Reward Structure:** Progress-based scoring for completed objectives

## 🔧 Installation
```bash
pip install -e .
```

## 🏆 Achievements
- ✅ Spec Compliance: Follows strict OpenEnv step()/reset() API
- ✅ Solvability: Oracle script proves perfect score achievable
- ✅ Robustness: Self-correction loop for invalid AI responses
- ✅ Efficiency: Free-tier AI model deployment via Hugging Face
