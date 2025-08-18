#!/usr/bin/env python3
"""
UI检查框架使用示例
演示如何使用YAML配置检查不同类型的UI组件
"""

from ui_checker import UIChecker, ComponentData
import json


def create_stock_card_examples():
    """创建股票卡片组件示例"""
    
    # 正确的股票卡片示例
    good_stock_card = ComponentData(
        name="good_stock_card",
        type="stock_card",
        props={
            "stock_code": "000001",
            "stock_name": "平安银行", 
            "current_price": 15.50,
            "price_change": 0.32,
            "change_percentage": 2.11
        },
        styles={
            "width": "250px",
            "height": "120px",
            "color": "#52c41a",  # 绿色表示上涨
            "background_color": "#ffffff",
            "border": "1px solid #d9d9d9",
            "border_radius": "6px",
            "focus": {
                "outline": "2px solid #1890ff"
            }
        },
        metadata={
            "data_timestamp": "2025-01-15T10:30:00"
        }
    )
    
    # 有问题的股票卡片示例
    bad_stock_card = ComponentData(
        name="bad_stock_card", 
        type="stock_card",
        props={
            "stock_code": "1",  # 错误：不是6位数字
            "stock_name": "这是一个很长的股票名称超过了限制",  # 错误：名称过长
            "current_price": 15.50,
            "price_change": 0.32,
            "change_percentage": 2.11
        },
        styles={
            "width": "100px",  # 错误：宽度太小
            "height": "50px",  # 错误：高度太小
            "color": "#f5222d",  # 错误：价格上涨应该是绿色，不是红色
            "background_color": "#ffffff"
            # 缺少focus样式
        }
    )
    
    return [good_stock_card, bad_stock_card]


def create_trading_panel_example():
    """创建交易面板组件示例"""
    
    trading_panel = ComponentData(
        name="trading_panel",
        type="trading_panel", 
        props={
            "stock_selection": "000001",
            "operation_type": "buy",
            "quantity": 1000,
            "price": 15.50,
            "order_type": "limit",
            "aria-label": "股票交易面板"
        },
        styles={
            "padding": "16px",
            "border_radius": "8px",
            "background_color": "#ffffff"
        },
        metadata={
            "available_balance": 50000,
            "portfolio_value": 200000,
            "same_stock_positions": 2
        }
    )
    
    return trading_panel


def create_modal_example():
    """创建模态对话框组件示例"""
    
    modal = ComponentData(
        name="confirmation_modal",
        type="modal",
        props={
            "title": "确认交易",
            "aria-modal": "true"
        },
        styles={
            "max_width": "520px",
            "border_radius": "8px",
            "box_shadow": "0 6px 16px rgba(0, 0, 0, 0.12)",
            "backdrop_color": "rgba(0, 0, 0, 0.45)"
        },
        children=[
            ComponentData(
                name="modal_header",
                type="div",
                props={"close_button": True}
            ),
            ComponentData(
                name="modal_footer", 
                type="div",
                props={"action_buttons": True}
            )
        ]
    )
    
    return modal


def create_page_examples():
    """创建页面级别检查示例"""
    
    # 仪表板页面
    dashboard_page = {
        "type": "dashboard_page",
        "components": [
            "portfolio_summary",
            "market_overview", 
            "quick_actions"
        ],
        "breakpoints": ["576px", "768px", "992px", "1200px"],
        "load_time": 2.5,
        "auto_refresh_interval": 30
    }
    
    # 缺少必需组件的仪表板页面
    incomplete_dashboard = {
        "type": "dashboard_page", 
        "components": [
            "market_overview"  # 缺少portfolio_summary和quick_actions
        ],
        "breakpoints": ["768px"]  # 缺少其他断点
    }
    
    return dashboard_page, incomplete_dashboard


def run_comprehensive_check():
    """运行全面的UI检查示例"""
    
    print("🔍 开始UI设计符合性检查...")
    print("=" * 60)
    
    try:
        # 初始化检查器
        checker = UIChecker(
            schema_file="ui_validation_schema.yaml",
            rules_file="component_check_rules.yaml"
        )
        
        # 准备测试组件
        stock_cards = create_stock_card_examples()
        trading_panel = create_trading_panel_example()
        modal = create_modal_example()
        
        all_components = stock_cards + [trading_panel, modal]
        
        # 准备页面数据
        good_page, bad_page = create_page_examples()
        
        print("📋 检查组件列表:")
        for comp in all_components:
            print(f"  - {comp.name} ({comp.type})")
        print()
        
        # 运行检查 - 好的页面
        print("✅ 检查正确配置的页面...")
        checker.run_checks(all_components, good_page)
        
        good_results = len([r for r in checker.results if not r.passed])
        print(f"发现 {good_results} 个问题")
        print()
        
        # 运行检查 - 有问题的页面
        print("❌ 检查有问题的页面...")
        checker.run_checks(all_components, bad_page)
        
        # 生成详细报告
        print("📊 生成检查报告...")
        print("=" * 60)
        
        # JSON格式报告
        json_report = json.loads(checker.generate_report("json"))
        
        print("📈 检查摘要:")
        summary = json_report["summary"]
        print(f"  总检查项: {summary['total_checks']}")
        print(f"  通过: {summary['passed_checks']}")
        print(f"  失败: {summary['failed_checks']}")
        print(f"  成功率: {summary['success_rate']}")
        print()
        
        print("🚨 问题分布:")
        print(f"  错误: {summary['error_count']}")
        print(f"  警告: {summary['warning_count']}")
        print(f"  信息: {summary['info_count']}")
        print()
        
        # 显示具体问题
        if summary['error_count'] > 0:
            print("🔴 错误详情:")
            for error in json_report["results_by_severity"]["errors"]:
                print(f"  - [{error['component']}] {error['message']}")
                if error['suggestion']:
                    print(f"    💡 建议: {error['suggestion']}")
            print()
            
        if summary['warning_count'] > 0:
            print("🟡 警告详情:")
            for warning in json_report["results_by_severity"]["warnings"][:5]:  # 只显示前5个
                print(f"  - [{warning['component']}] {warning['message']}")
            if len(json_report["results_by_severity"]["warnings"]) > 5:
                print(f"  ... 还有 {len(json_report['results_by_severity']['warnings']) - 5} 个警告")
            print()
            
        # 按组件分组显示
        print("📦 按组件分组的问题:")
        for component, issues in json_report["results_by_component"].items():
            failed_issues = [i for i in issues if not i["passed"]]
            if failed_issues:
                print(f"  {component}: {len(failed_issues)} 个问题")
                for issue in failed_issues[:3]:  # 只显示前3个
                    print(f"    - {issue['message']}")
                if len(failed_issues) > 3:
                    print(f"    ... 还有 {len(failed_issues) - 3} 个问题")
        print()
        
        # Markdown报告保存
        md_report = checker.generate_report("markdown")
        with open("ui_check_report.md", "w", encoding="utf-8") as f:
            f.write(md_report)
        print("📄 详细报告已保存到: ui_check_report.md")
        
        # JSON报告保存
        with open("ui_check_report.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(json_report, indent=2, ensure_ascii=False))
        print("📄 JSON报告已保存到: ui_check_report.json")
        
    except Exception as e:
        print(f"❌ 检查过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_custom_rules():
    """演示自定义规则的使用"""
    
    print("\n" + "=" * 60)
    print("🎯 演示自定义规则检查")
    print("=" * 60)
    
    # 创建一个违反多个自定义规则的组件
    problematic_component = ComponentData(
        name="problematic_stock_card",
        type="stock_card",
        props={
            "stock_code": "ABCD12",  # 错误格式
            "stock_name": "平安银行",
            "current_price": 15.50,
            "price_change": -0.32,  # 下跌
            "change_percentage": -2.11
        },
        styles={
            "width": "150px",  # 太小
            "height": "80px",   # 太小
            "color": "#52c41a",  # 错误：下跌应该是红色
            "background_color": "#52c41a"  # 错误：与文字颜色相同
        },
        metadata={
            "data_timestamp": "2025-01-10T10:30:00"  # 数据过期
        }
    )
    
    try:
        checker = UIChecker(
            schema_file="ui_validation_schema.yaml",
            rules_file="component_check_rules.yaml"
        )
        
        checker.run_checks([problematic_component])
        
        print("发现的具体问题:")
        for result in checker.results:
            if not result.passed:
                print(f"  🔍 {result.name}")
                print(f"     规则: {result.rule}")
                print(f"     消息: {result.message}")
                if result.expected and result.actual:
                    print(f"     期望: {result.expected}")
                    print(f"     实际: {result.actual}")
                print()
                
    except Exception as e:
        print(f"❌ 自定义规则检查失败: {e}")


if __name__ == "__main__":
    # 运行全面检查
    run_comprehensive_check()
    
    # 演示自定义规则
    demonstrate_custom_rules()
    
    print("\n🎉 UI检查框架演示完成！")
    print("💡 提示: 查看生成的报告文件了解详细结果")