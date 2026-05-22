# CBB Edge - Elite UAT Workflow

## Overview

A repeatable, automated User Acceptance Testing (UAT) workflow that evaluates the application from two expert perspectives:
1. **Elite Fantasy Baseball Manager** - Deep domain expertise in fantasy baseball strategy
2. **Quant Trading & Sabermetrics Expert** - Deep expertise in quantitative analysis, edge detection, and statistical modeling

## Workflow Architecture

```
├──────────────────────────────────────────────────────────────────────┐
│  Phase 1: Environment Setup & Authentication                    │
│  - Start dev server                                              │
│  - Authenticate via API key                                      │
│  - Verify data freshness                                         │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
├──────────────────────────────────────────────────────────────────────┐
│  Phase 2: Elite Fantasy Manager Evaluation                       │
│  - War Room feature assessment                                   │
│  - Roster optimization evaluation                                │
│  - Waiver wire analysis                                          │
│  - Streaming recommendations                                     │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
├──────────────────────────────────────────────────────────────────────┐
│  Phase 3: Quant Trading & Sabermetrics Evaluation                 │
│  - Edge calculation verification                                 │
│  - CLV (Closing Line Value) analysis                             │
│  - Statistical model validation                                  │
│  - Data quality assessment                                       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
├──────────────────────────────────────────────────────────────────────┐
│  Phase 4: Report Generation & Action Items                        │
│  - Score calculation                                             │
│  - Issue prioritization                                          │
│  - Improvement recommendations                                   │
│  - Executive summary                                             │
└──────────────────────────────────────────────────────────────────────┘
```

## Execution Frequency

- **Daily**: Automated smoke tests (5 minutes)
- **Weekly**: Full UAT evaluation (30 minutes)
- **Monthly**: Deep dive assessment (2 hours)
- **Pre-release**: Complete evaluation with all features

## Required Tools

1. **Playwright** - Browser automation
2. **Devtools MCP** - Browser introspection and network monitoring
3. **Custom evaluators** - Domain-specific assessment logic

## Success Criteria

- All critical paths functional
- No P0/P1 issues identified
- Fantasy manager score ≥ 8.0/10
- Quant trading score ≥ 8.0/10
- Performance metrics within acceptable ranges
