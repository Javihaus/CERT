#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CERT Baseline Measurement for Claude Sonnet 4.5 - Direct Measurement

This script performs the baseline measurement directly using Claude Code's
ability to self-measure during the conversation.

Methodology matches CERT paper exactly:
- Behavioral Consistency (C): 20 trials
- Performance Baseline (mu, sigma): 15 prompts x 5 samples
- Context Propagation (gamma): 5 prompt pairs
"""

# I will perform this measurement interactively by responding to the prompts
# and having them scored objectively.

print("""
CERT BASELINE MEASUREMENT - CLAUDE SONNET 4.5
==============================================

This is a direct measurement where Claude Sonnet 4.5 (me) will respond to
the standardized prompts and score them objectively.

METHODOLOGY:
1. Behavioral Consistency (C): 20 identical prompts
2. Performance Baseline (mu, sigma): 15 prompts x 5 samples = 75 responses
3. Context Propagation (gamma): 5 prompt pairs (10 responses)

Total: 105 responses

Let's begin...
""")
