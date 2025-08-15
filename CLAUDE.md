# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an A-share (Chinese stock market) analysis system that fetches trading data and performs AI-powered analysis using the DeepSeek LLM API. The system provides comprehensive stock analysis including historical data retrieval, event analysis, technical analysis, and trading recommendations.

## Code Rules
please refer to prompts/arch.md

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **Core Application (`src/main.py`)**: Command-line interface and workflow orchestration
- **Data Layer (`src/data/`)**: Stock data fetching with caching, supports both real data (akshare) and mock data
- **Analysis Layer (`src/analysis/`)**: LLM-powered analysis with fallback to basic technical analysis
- **Cache Layer (`src/cache/`)**: File-based caching system for API responses and trading data
- **Configuration (`src/config/`)**: Centralized settings management with environment variable support
- **Reports (`reports/`)**: Generated Markdown analysis reports with timestamps
- **Articles (`articles/`)**: Input articles for analysis and generated analysis results

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

### Running the Application
```bash
# Quick start script (activates venv automatically)
./run.sh

# Manual execution with various options
python src/main.py --stock 000001      # Analyze specific stock
python src/main.py --verbose           # Verbose output
python src/main.py --list-reports      # List historical reports
python src/main.py --show-report filename.md  # Display specific report
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run tests with coverage
python -m pytest tests/ -v --cov=src

# Run specific test categories
python -m pytest tests/unit/           # Unit tests only
python -m pytest tests/integration/    # Integration tests only
python -m pytest tests/functional/     # End-to-end tests
```

### Linting and Code Quality
The project uses pytest for testing. No specific linting configuration found - consider adding flake8, black, or similar tools.

## Key Configuration

- **API Keys**: DeepSeek API key stored in `src/config/settings.py` and can be overridden via `DEEPSEEK_API_KEY` environment variable
- **Cache Settings**: 7-day cache expiry by default, stored in `cache/` directory
- **Data Sources**: Uses akshare library for real A-share data fetching
- **Reports**: Generated in `reports/` with timestamp-based naming

## Important Files and Patterns

- **Settings Management**: All configuration centralized in `src/config/settings.py` using dataclasses
- **Cache Keys**: Follow pattern `trading_data_{stock_code}` for consistency
- **Report Naming**: Format: `{stock_code}_{YYYYMMDD}_{HHMMSS}.md`
- **Error Handling**: System includes fallback analysis when LLM API fails
- **Logging**: Configured in main.py with INFO level by default, DEBUG with --verbose flag

## Analysis Workflow

1. **Data Fetching**: Check cache first, then fetch from akshare API
2. **LLM Analysis**: Send data to DeepSeek API for comprehensive analysis
3. **Fallback Analysis**: Basic technical analysis if LLM fails
4. **Report Generation**: Save results as Markdown in `reports/` directory
5. **Cache Management**: Store API responses and processed data

## Testing Strategy

- **Unit Tests**: Individual component testing (cache, config, data fetchers, report manager)
- **Integration Tests**: System-level integration testing
- **Functional Tests**: End-to-end workflow testing
- **Test Fixtures**: Sample data in `tests/fixtures/sample_data.py`

## Special Considerations

- The system includes specialized prompts for "八骏" (Bajun) stock analysis in `articles/prompts/bajun.md`
- Stock code mapping and analysis rules are defined for different market sectors
- All reports are generated in Chinese for the target A-share market
- The system supports both individual stock analysis and market-wide analysis
