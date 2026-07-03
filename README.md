# lucid

**A playable, learned dream — and an agent that grows up inside it.**

A lucid dream is a dream you can control. `lucid` learns one from raw CoinRun
video: a Genie-style world model trained with no action labels, whose latent
action codes make the dream playable with a keyboard. A DreamerV3-style
actor-critic is then trained *entirely inside the dream* — it never touches the
real game — and is finally deployed in real CoinRun to measure how well a
policy learned in imagination transfers to reality.

The headline result this project builds to: **a real-game score from an agent
that never trained on the real game.**

See [PLAN.md](PLAN.md) for the full implementation plan: build stages and
gates, metrics, repo structure, timeline, and de-risking.
