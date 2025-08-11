"""
报告管理器单元测试
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.analysis.report_manager import ReportManager


@pytest.fixture
def temp_reports_dir():
    """临时报告目录"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def report_manager(temp_reports_dir):
    """报告管理器实例"""
    return ReportManager(reports_dir=temp_reports_dir)


@pytest.fixture
def sample_analysis():
    """样本分析结果"""
    return {
        'events_analysis': '这是事件分析内容',
        'trend_analysis': '这是技术分析内容',
        'trading_advice': '这是交易建议内容',
        'analysis_time': '2024-01-01T12:00:00'
    }


class TestReportManager:
    """报告管理器测试类"""
    
    def test_format_analysis_to_markdown(self, report_manager, sample_analysis):
        """测试分析结果格式化为Markdown"""
        markdown_content = report_manager.format_analysis_to_markdown(
            sample_analysis, "000001"
        )
        
        # 验证Markdown格式
        assert "# 000001 - 股票分析报告" in markdown_content
        assert "## 📊 重大事件分析" in markdown_content
        assert "## 📈 技术面分析" in markdown_content
        assert "## 💡 交易建议" in markdown_content
        assert "## ⚠️ 免责声明" in markdown_content
        
        # 验证内容包含
        assert "这是事件分析内容" in markdown_content
        assert "这是技术分析内容" in markdown_content
        assert "这是交易建议内容" in markdown_content
    
    def test_save_report(self, report_manager, sample_analysis):
        """测试保存报告"""
        markdown_content = report_manager.format_analysis_to_markdown(
            sample_analysis, "000001"
        )
        
        report_path = report_manager.save_report(markdown_content, "000001")
        
        # 验证文件保存
        assert Path(report_path).exists()
        assert "000001_" in report_path
        assert report_path.endswith(".md")
        
        # 验证文件内容
        with open(report_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == markdown_content
    
    def test_get_latest_report(self, report_manager, sample_analysis):
        """测试获取最新报告"""
        # 初始状态无报告
        latest = report_manager.get_latest_report("000001")
        assert latest is None
        
        # 保存报告
        markdown_content = report_manager.format_analysis_to_markdown(
            sample_analysis, "000001"
        )
        report_manager.save_report(markdown_content, "000001")
        
        # 获取最新报告
        latest = report_manager.get_latest_report("000001")
        assert latest == markdown_content
    
    def test_list_reports(self, report_manager, sample_analysis):
        """测试列出报告"""
        # 初始状态无报告
        reports = report_manager.list_reports()
        assert len(reports) == 0
        
        # 保存几个报告
        for stock_code in ["000001", "000002"]:
            markdown_content = report_manager.format_analysis_to_markdown(
                sample_analysis, stock_code
            )
            report_manager.save_report(markdown_content, stock_code)
        
        # 列出所有报告
        all_reports = report_manager.list_reports()
        assert len(all_reports) == 2
        
        # 列出特定股票报告
        stock_reports = report_manager.list_reports("000001")
        assert len(stock_reports) == 1
        assert "000001_" in stock_reports[0]
    
    def test_clean_old_reports(self, report_manager, sample_analysis):
        """测试清理旧报告"""
        # 保存报告
        markdown_content = report_manager.format_analysis_to_markdown(
            sample_analysis, "000001"
        )
        report_path = report_manager.save_report(markdown_content, "000001")
        
        # 修改文件时间为很久以前
        import os
        import time
        old_time = time.time() - (31 * 24 * 60 * 60)  # 31天前
        os.utime(report_path, (old_time, old_time))
        
        # 清理30天前的报告
        cleaned_count = report_manager.clean_old_reports(keep_days=30)
        assert cleaned_count == 1
        
        # 验证文件已删除
        assert not Path(report_path).exists()
    
    def test_format_section(self, report_manager):
        """测试章节格式化"""
        # 测试空内容
        result = report_manager._format_section("")
        assert result == "*暂无内容*"
        
        # 测试普通文本
        result = report_manager._format_section("这是普通文本")
        assert result == "这是普通文本"
        
        # 测试已有Markdown格式
        markdown_text = "**这是粗体**\n- 这是列表"
        result = report_manager._format_section(markdown_text)
        assert result == markdown_text
    
    def test_format_note(self, report_manager):
        """测试备注格式化"""
        # 测试空备注
        result = report_manager._format_note("")
        assert result == ""
        
        # 测试有备注
        result = report_manager._format_note("这是备注")
        assert "> **说明**: 这是备注" in result