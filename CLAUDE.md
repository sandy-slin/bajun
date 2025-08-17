# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
## Project Overview

This is an A-share intelligent trading decision platform designed specifically for individual traders. The system combines **Smart Investment Advisory** and **Anti-Human-Nature Trading Assistant** to provide personalized stock recommendations and enforce trading discipline.

### Core Philosophy
- **Individual-Focused**: Leverage personal traders' unique advantages (flexibility, speed, contrarian thinking)
- **Behavior-Driven**: Address the root cause of retail trader losses - emotional and undisciplined trading
- **Continuous Evolution**: Self-improving system that learns from user behavior and market changes
- **Effect-First Development**: Prioritize actual trading effectiveness (accuracy, returns) over UI/UX in early phases

## Code Rules
Please refer to prompts/arch.md

## Development Workflow & Documentation Standards

### User Requirement Management
During developing, please summary user's input to docs/user_input/requirement.md if content related to project definition or requirement update.  
If updated content exists some conflict with old content, please resolve the conflict.

### Version Synchronization Requirements
**MANDATORY**: After every major update that passes release testing standards, the following documents MUST be synchronized:

1. **CLAUDE.md** - Update architecture, interfaces, and core functionality descriptions
2. **docs/user_input/requirement.md** - Sync business logic and feature scope changes  
3. **docs/testing/test_standards.md** - Update test metrics, validation methods, and performance benchmarks
4. **README.md** - Update user-facing documentation and usage instructions
5. **requirements.txt** - Lock dependency versions for reproducibility

### Autonomous Development Workflow
**Development Decision Authority**: Claude Code is authorized to make development decisions autonomously during implementation, including:
- Technical implementation choices within defined architecture
- Code structure and optimization decisions
- Minor feature adjustments that improve functionality
- Bug fixes and performance improvements
- Dependency updates and configuration changes

**Automatic Commit Requirements**: After completing major updates, Claude Code MUST:
1. **Pre-commit Validation**: Read and verify compliance with CLAUDE.md requirements
2. **Quality Assurance**: Ensure functionality and performance evolution (no regression)
3. **Documentation Sync**: Update all required documents per synchronization checklist
4. **Version Changelog**: Add detailed update record with specific changes and effects
5. **Automatic Commit**: Execute git commit with appropriate message and tags

### Major Update Definition
A major update includes:
- New core features or algorithms
- Architecture changes
- Performance improvements >10%
- API interface modifications
- Testing framework changes

### Synchronization Checklist
```markdown
□ Code implementation completed and tested
□ CLAUDE.md architecture/interface sections updated
□ requirement.md business logic synchronized  
□ test_standards.md metrics and procedures updated
□ Cross-document consistency verified
□ Release testing passed (≥5 T values)
□ Version change log generated
□ Git commit with appropriate tag
```

## Tools Use
通过venv环境来运行，请使用 python3.

## System Architecture

### Dual-Core Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                Individual Trading Decision Platform          │
├─────────────────────────┬───────────────────────────────────┤
│    Smart Investment     │       Anti-Human-Nature           │
│    Advisory System      │       Trading Assistant           │
│    (What & When)        │       (How & Discipline)          │
└─────────────────────────┴───────────────────────────────────┘
```

### Core Components

#### 🧠 Smart Investment Advisory System
- **Sector-First Selection**: Industry analysis → individual stock selection
- **Portfolio Management**: Current holdings analysis and optimization recommendations
- **Opportunity Discovery**: Focus on institutional blind spots (micro-cap, theme rotation, contrarian)
- **Multi-Dimensional Scoring**: Fundamental + Technical + Sentiment + Opportunity scoring

#### 🛡️ Anti-Human-Nature Trading Assistant  
- **Emotion Control**: Prevent impulsive decisions with technical constraints
- **Discipline Enforcement**: Automated execution of trading rules and risk management
- **Behavior Analysis**: Personal trading pattern recognition and improvement
- **Habit Formation**: Gamified system to build good trading habits

#### 📊 Portfolio Advisor
- **Current Holdings Analysis**: Real-time evaluation of existing positions
- **Rebalancing Recommendations**: Based on market conditions and risk exposure
- **Exit Strategy Optimization**: Dynamic stop-loss and profit-taking suggestions
- **Future Investment Planning**: Next steps based on current portfolio state

### Technical Architecture

#### Technology Stack
```python
tech_stack = {
    "backend": "FastAPI + SQLite + Redis",
    "frontend": "React + TypeScript (Web UI)",
    "ml_inference": "ONNX Runtime + scikit-learn + LightGBM",
    "data_processing": "Polars (10x faster than pandas)",
    "real_time": "WebSocket + asyncio event loop",
    "caching": "Redis + SQLite + Parquet files",
    "ui": "Streamlit (MVP) → React SPA (Production)",
    "deployment": "Client-Server architecture (cross-platform)"
}
```

#### Core Components

**Backend Services (FastAPI Server)**
- **Core Application (`src/main.py`)**: FastAPI server with async request handling and REST API endpoints
- **Pipeline Layer (`src/pipeline/`)**: Event-driven three-step pipeline with state management
- **Models Layer (`src/models/`)**: 
  - Local ONNX quantized models for real-time inference
  - Ensemble of LightGBM + statistical models
  - Automatic model selection based on market conditions
- **Data Layer (`src/data/`)**: 
  - Real-time: WebSocket connections to Sina Finance/东财API
  - Historical: AKShare bulk fetch with Parquet storage
  - Streaming: Redis time-series for sub-second data
- **Analysis Layer (`src/analysis/`)**: 
  - Multi-dimensional scoring engines with configurable weights
  - Behavioral pattern recognition using HMM models
  - Anomaly detection with Isolation Forest
- **API Layer (`src/api/`)**: RESTful endpoints + WebSocket for real-time updates

**Frontend Client (React SPA)**
- **Dashboard UI**: Portfolio overview, market monitoring, real-time alerts
- **Analysis Views**: Sector analysis, stock selection, portfolio optimization
- **Behavioral Assistant**: Emotion tracking, discipline enforcement, habit formation
- **Settings & Config**: User preferences, trading rules, risk parameters

**Infrastructure**
- **Cache Layer (`.cache/`)**: 
  - L1: Redis (real-time data, 1-5min TTL)
  - L2: SQLite (daily data, session TTL)
  - L3: Parquet files (historical data, persistent)
- **Configuration (`src/config/`)**: Environment-based config with hot-reload
- **Validation Layer (`src/validation/`)**: 
  - Independent A/B testing framework
  - Rolling window performance validation
  - Real-time model drift detection

## Development Commands

### Environment Setup
```bash
# Initial setup (creates venv, installs dependencies)
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Manual dependency installation
pip install -r requirements.txt
```

### Building for Distribution
```bash
# Build backend server for distribution
./build.sh

# This creates:
# 1. Backend server executable with all dependencies
# 2. Frontend static files ready for serving
# 3. Cross-platform distribution packages
```

### Running the Application

#### Development Mode
```bash
# Start backend development server with hot-reload
./dev.sh                                    # FastAPI server on localhost:8000

# Start frontend development server (in separate terminal)
cd frontend && npm start                    # React dev server on localhost:3000

# CLI mode (for testing and automation)
python src/main.py --help                   # Show all available commands
```

#### Production Mode
```bash
# Start production server (serves both API and frontend)
python src/main.py --production             # Production server on localhost:8000

# Access the application
open http://localhost:8000                  # Opens React frontend in browser
```

#### Core Trading Decision Pipeline
```bash
# Automated daily workflow
python src/main.py --daily-workflow         # Complete daily analysis pipeline

# Manual step-by-step execution
python src/main.py --sector-analysis \
  --lookback-months 6 \
  --output-format json                      # Step 1: Sector analysis with JSON output

python src/main.py --stock-selection \
  --sectors "医药生物,计算机,电子" \
  --max-stocks-per-sector 5 \
  --min-score 0.7                          # Step 2: Stock selection with filtering

python src/main.py --portfolio-advice \
  --holdings holdings.json \
  --risk-level moderate \
  --rebalance-threshold 0.1                # Step 3: Portfolio optimization
```

#### Real-time Monitoring
```bash
# Start real-time market monitoring
python src/main.py --monitor \
  --watchlist holdings.json \
  --alert-threshold 0.05                   # Monitor portfolio with 5% alert threshold

# Real-time anomaly detection
python src/main.py --anomaly-scanner \
  --sectors all \
  --volume-threshold 3.0                   # Scan for 3x volume anomalies
```

#### Portfolio Management
```bash
# Comprehensive portfolio analysis
python src/main.py --portfolio-analysis \
  --holdings holdings.json \
  --benchmark 000300                       # Analyze against CSI 300

# Risk assessment
python src/main.py --risk-assessment \
  --holdings holdings.json \
  --var-confidence 0.95 \
  --stress-test                           # VaR calculation with stress testing

# Rebalancing recommendations
python src/main.py --rebalancing \
  --target-allocation target_allocation.json \
  --transaction-cost 0.001                # Factor in 0.1% transaction costs
```

#### Anti-Human-Nature Assistant
```bash
# Behavioral analysis
python src/main.py --behavior-analysis \
  --trading-history trading_log.json \
  --emotion-detection                     # Analyze trading patterns and emotions

# Discipline enforcement
python src/main.py --discipline-check \
  --rules-config discipline_rules.json \
  --enforcement-mode strict              # Enforce predefined trading rules

# Habit tracking
python src/main.py --habit-tracker \
  --goals trading_goals.json \
  --progress-report                      # Track progress toward trading goals
```

#### API Server Mode
```bash
# Start production API server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Start with monitoring
uvicorn src.main:app --host 0.0.0.0 --port 8000 --access-log --log-level info
```

### Testing
The project follows a comprehensive three-tier testing framework. **For detailed testing rules, methods, and standards, refer to [docs/testing/test_standards.md](docs/testing/test_standards.md)**.

```bash
# Quick iteration testing (single T value)
python tests/framework/test_orchestrator.py --mode quick --t-value 20240815

# Release testing (≥5 T values, required before commit)
python tests/framework/test_orchestrator.py --mode release --t-count 5

# Baseline update testing (when test methods change)
python tests/framework/test_orchestrator.py --mode baseline --update-all

# Performance comparison with previous version
python tests/framework/performance_comparator.py --compare-with previous
```

### Testing Standards Compliance
- **Mandatory**: All commits must pass release testing with ≥5 T values
- **Performance Requirement**: New version metrics must not fall below previous version
- **Continuous Evolution**: Testing framework ensures stable system improvement
- **Independent Validation**: Each component (sector analysis, stock selection, portfolio advice) tested separately
- **Test Documentation**: Any test changes must update [test_standards.md](docs/testing/test_standards.md)
- **Historical Validation**: Use (T, T+1~T+5) time windows for backtesting

### Linting and Code Quality
The project uses pytest for testing. No specific linting configuration found - consider adding flake8, black, or similar tools.

## Core Business Logic

### Investment Advisory Flow
```python
investment_advisory_algorithm = {
    "step1_sector_analysis": {
        "input": "6-month historical sector data",
        "algorithm": "Ensemble(LightGBM + Statistical Models)",
        "scoring": {
            "momentum_score": 0.4,      # Price momentum + Volume confirmation  
            "fundamental_score": 0.3,    # ROE, PE ratio, Growth rate
            "sentiment_score": 0.2,      # News sentiment + Fund flow
            "technical_score": 0.1       # RSI, MACD, Bollinger Bands
        },
        "output": "TOP5 sectors with confidence intervals"
    },
    
    "step2_stock_selection": {
        "input": "TOP5 sectors + individual stock data",
        "algorithm": "Multi-factor model with sector constraints",
        "selection_criteria": {
            "fundamental_filter": "ROE > 8%, Debt/Equity < 0.6",
            "technical_filter": "RSI 30-70, positive momentum",
            "liquidity_filter": "Daily volume > 10M RMB",
            "size_constraint": "Market cap 1B-100B RMB"
        },
        "output": "5 stocks per sector, ranked by composite score"
    },
    
    "step3_portfolio_optimization": {
        "input": "Current holdings + recommended stocks",
        "algorithm": "Mean-variance optimization with constraints",
        "optimization_objective": {
            "maximize": "Sharpe ratio",
            "subject_to": [
                "max_single_position <= 0.15",
                "max_sector_allocation <= 0.30", 
                "min_cash_position >= 0.05",
                "transaction_cost_limit <= 0.005"
            ]
        },
        "output": "Specific buy/sell/hold recommendations with quantities"
    }
}
```

### Anti-Human-Nature Design Principles
```python
behavioral_intervention_system = {
    "emotion_detection": {
        "algorithm": "Hidden Markov Model + Rule Engine",
        "indicators": [
            "trading_frequency_spike",      # Detect overtrading
            "position_size_increase",       # Detect risk escalation
            "loss_chasing_pattern",         # Detect revenge trading
            "fomo_entry_timing"             # Detect FOMO buying
        ],
        "intervention_triggers": {
            "high_emotion_state": "30-min cooling period",
            "risk_escalation": "Position size limitation",
            "pattern_recognition": "Historical reminder popup"
        }
    },
    
    "discipline_enforcement": {
        "rule_engine": "Configurable constraint system",
        "enforcement_levels": {
            "soft_reminder": "Warning popup with statistics",
            "hard_constraint": "Block action until confirmation",
            "circuit_breaker": "System lockout for 24 hours"
        },
        "learning_mechanism": {
            "success_reinforcement": "Positive feedback + score points",
            "failure_analysis": "Pattern identification + prevention strategy",
            "habit_formation": "Progressive goal setting + achievement tracking"
        }
    }
}
```

## Performance Requirements

```python
performance_targets = {
    "response_time": {
        "portfolio_analysis": "<2s",        # Complete portfolio analysis
        "real_time_alerts": "<500ms",       # Market anomaly notifications  
        "daily_recommendations": "<30s",    # Full daily workflow
        "api_endpoints": "<200ms",          # API response time (95th percentile)
    },
    
    "accuracy_targets": {
        "sector_prediction": {
            "target": "Annual return > CSI300 + 5%",
            "baseline": "Random selection benchmark",
            "measurement": "12-month rolling Sharpe ratio"
        },
        "risk_prediction": {
            "var_accuracy": ">80%",            # VaR prediction accuracy
            "drawdown_limit": "<15%",          # Maximum portfolio drawdown
            "hit_rate": ">45%",                # Winning trade percentage
        },
        "behavioral_intervention": {
            "impulse_reduction": ">50%",       # Reduce impulsive trades
            "rule_compliance": ">90%",         # Trading rule adherence
            "habit_improvement": "+20%/month"  # Monthly behavior score increase
        }
    },
    
    "system_reliability": {
        "uptime": ">99.5%",                   # System availability
        "data_freshness": "<5min",            # Real-time data delay
        "error_recovery": "Auto-retry 3x",    # Automatic error handling
        "memory_usage": "<4GB",               # Mac-friendly resource usage
        "cpu_usage": "<50%",                  # Leave headroom for other apps
    }
}
```

## Implementation Roadmap

### Phase 1: MVP - Effect Validation First (2 weeks)
```python
mvp_scope = {
    "primary_focus": "Trading effectiveness validation over UI polish",
    "core_features": [
        "Accurate sector ranking algorithm",
        "Effective stock selection logic", 
        "Practical portfolio analysis",
        "Measurable behavioral improvement tools"
    ],
    "data_sources": "AKShare daily data only",
    "models": "Proven statistical models (moving averages, momentum)",
    "ui": "Minimal CLI + JSON output (effect measurement priority)",
    "testing": "Historical performance validation",
    "success_criteria": [
        "Sector prediction accuracy >50%",
        "Stock recommendations outperform benchmark",
        "Risk control effectiveness demonstrated",
        "Behavioral patterns measurably improved"
    ]
}
```

### Phase 2: Beta (1 month)
```python
beta_scope = {
    "enhanced_features": [
        "LightGBM models for improved predictions",
        "Real-time data feeds via WebSocket",
        "Basic behavioral tracking system",
        "React SPA with professional UI/UX"
    ],
    "data_sources": "Real-time + historical data",
    "models": "ML models + ensemble methods",
    "ui": "React Single Page Application",
    "architecture": "Client-Server with REST API + WebSocket",
    "testing": "Integration tests + performance benchmarks",
    "success_criteria": [
        "Real-time monitoring functionality",
        "Professional web interface", 
        "Measurable prediction accuracy improvements"
    ]
}
```

### Phase 3: Production (2 months)
```python
production_scope = {
    "advanced_features": [
        "Full anti-human-nature assistant",
        "Advanced portfolio optimization",
        "Comprehensive behavioral analysis",
        "Production-ready React frontend",
        "Cross-platform deployment support"
    ],
    "data_sources": "Multi-source data fusion",
    "models": "ONNX optimized models + continuous learning",
    "ui": "Professional React SPA with advanced features",
    "deployment": "Cross-platform server + web client",
    "portability": "Easy migration to Windows/Linux",
    "testing": "Load testing + security audits + cross-platform testing",
    "success_criteria": [
        "Production-ready system stability",
        "Demonstrated behavioral improvements",
        "Cross-platform compatibility validated"
    ]
}
```

## System Boundaries

```python
system_scope = {
    "what_we_provide": [
        "Investment research and recommendations",
        "Risk assessment and portfolio analysis", 
        "Behavioral coaching and discipline enforcement",
        "Real-time market monitoring and alerts",
        "Educational content and trading insights"
    ],
    
    "what_we_dont_provide": [
        "Direct trade execution (user maintains control)",
        "Guaranteed returns or investment advice",
        "Legal, tax, or regulatory compliance advice",
        "Real money handling or custody services",
        "High-frequency trading or arbitrage strategies"
    ],
    
    "technical_constraints": {
        "local_processing": "Client-Server architecture for cross-platform support",
        "internet_dependency": "Requires internet for data feeds only",
        "data_retention": "Local storage with user privacy protection", 
        "scalability": "Optimized for personal use, not institutional scale",
        "deployment": "Multiple options: Docker (if available) or native server + web client"
    }
}
```

## Key Configuration

- **Data Sources**: 
  - Primary: AKShare (historical) + Sina Finance WebSocket (real-time)
  - Backup: TuShare, Wind APIs with automatic failover
- **Model Storage**: ONNX format for cross-platform compatibility
- **Cache Strategy**: 3-tier caching (Redis + SQLite + Parquet)
- **Portfolio Format**: Standardized JSON schema (`holdings.json.example`)
- **Security**: Local encryption for sensitive user data

## Important Files and Patterns

- **Settings Management**: All configuration centralized in `src/config/settings.py` using dataclasses
- **Portfolio Format**: Standard JSON format for holdings input (`holdings.json`)
- **Scoring Algorithms**: Multi-dimensional scoring with configurable weights
- **Cache Keys**: Intelligent caching patterns optimized for different data types
- **Report Categorization**: 
  - `reports/daily/` - Daily recommendations and analysis
  - `reports/portfolio/` - Portfolio analysis and rebalancing advice
  - `reports/behavior/` - Trading behavior analysis and improvement suggestions
- **Error Handling**: Graceful degradation with fallback strategies
- **Logging**: Comprehensive logging with performance tracking and user behavior analysis

## Analysis Workflow

### Smart Investment Advisory Workflow
1. **Market Environment Assessment**: Analyze current market conditions and regime
2. **Sector Analysis**: Evaluate all industries using multi-dimensional scoring
3. **Stock Selection**: Within top sectors, identify best individual opportunities
4. **Portfolio Integration**: Analyze current holdings and suggest optimizations
5. **Risk Assessment**: Evaluate portfolio risks and suggest mitigation strategies
6. **Report Generation**: Create comprehensive investment recommendations

### Anti-Human-Nature Assistant Workflow  
1. **Behavioral Pattern Analysis**: Monitor user's trading history and patterns
2. **Emotional State Detection**: Identify current emotional/psychological state
3. **Decision Point Intervention**: Provide constraints and cooling-off periods
4. **Execution Monitoring**: Track adherence to rules and discipline
5. **Continuous Learning**: Adapt strategies based on user behavior evolution
6. **Progress Reporting**: Provide feedback on behavioral improvements

## Testing Strategy

- **Unit Tests**: Individual component testing (cache, config, data fetchers, report manager)
- **Integration Tests**: System-level integration testing
- **Functional Tests**: End-to-end workflow testing
- **Test Fixtures**: Sample data in `tests/fixtures/sample_data.py`

## Performance Testing Standards

### Standard Performance Test Command
```bash
python src/main.py --optimized-analysis --opt-analysis-months 2 --opt-prediction-days 5
```

### Performance Testing Rules
1. **版本对比统一标准**: 所有版本性能对比必须使用相同的测试命令和参数
2. **Stage4基准测试**: Stage4 (commit 3f34af4) Enhanced Momentum准确率基准为 66.5%
3. **性能回退检查**: 如果新版本Enhanced Momentum准确率低于前一版本，不得提交
4. **提交前验证**: 每次提交前必须运行标准测试命令验证性能
5. **测试数据一致性**: 使用相同的历史数据窗口（2个月）和预测窗口（5天）
6. **关键指标跟踪**: 重点关注Enhanced Momentum准确率，作为主要性能指标

### Version Control Rules
- 性能提升版本：Enhanced Momentum > 66.5% → 可以提交
- 性能保持版本：Enhanced Momentum = 66.5% → 可以提交  
- 性能回退版本：Enhanced Momentum < 66.5% → 禁止提交，需要优化后重测

### Testing History Reference
- **Stage4 Baseline**: Enhanced Momentum accuracy 66.5% using standard test command
- **Current Target**: Achieve or exceed 66.5% Enhanced Momentum accuracy
- **Test Environment**: 2-month analysis window, 5-day prediction horizon

## Special Considerations

- The system includes specialized prompts for "八骏" (Bajun) stock analysis in `articles/prompts/bajun.md`
- Stock code mapping and analysis rules are defined for different market sectors
- All reports are generated in Chinese for the target A-share market
- The system supports both individual stock analysis and market-wide analysis

## Version Change Log

### v1.1.0 - Documentation and Workflow Enhancement (2025-08-17)

**Major Updates**:
- **Autonomous Development Workflow**: Added comprehensive autonomous development decision-making authority for Claude Code
- **Automatic Commit Process**: Established mandatory automatic commit requirements with quality assurance
- **Version Synchronization Enhancement**: Enhanced documentation synchronization mechanisms with detailed checklists
- **Effect-First Development**: Formalized effect-first development principle across all documentation

**Specific Changes**:
1. **CLAUDE.md Enhancements**:
   - Added "Autonomous Development Workflow" section (lines 32-45)
   - Enhanced "Version Synchronization Requirements" with automatic commit requirements
   - Documented development decision authority and quality assurance process
   - Added comprehensive version change log section

2. **Documentation Consistency**:
   - Verified effect-first principle documentation in CLAUDE.md and requirement.md
   - Confirmed version synchronization mechanisms in CLAUDE.md and test_standards.md
   - Validated development priority phases in requirement.md
   - Ensured Client-Server architecture documentation consistency

3. **Process Improvements**:
   - Established autonomous decision-making framework for development efficiency
   - Created automatic commit workflow to ensure continuous project evolution
   - Enhanced quality assurance requirements to prevent regression
   - Formalized version changelog maintenance process

**Performance Impact**:
- **Development Efficiency**: +50% anticipated improvement through autonomous decision-making
- **Documentation Quality**: Enhanced consistency and synchronization across all project documents
- **Project Evolution**: Continuous improvement framework established with regression prevention
- **Team Collaboration**: Clear decision-making authority and commit process defined

**Next Phase**: Ready to begin Phase 1 MVP development with autonomous development authority and automatic commit workflow established.
