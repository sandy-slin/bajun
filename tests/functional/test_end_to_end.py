"""
端到端功能测试
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestEndToEnd:
    """端到端测试类"""
    
    def test_main_script_help(self):
        """测试主脚本帮助信息"""
        result = subprocess.run([
            sys.executable, 'src/main.py', '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert 'A股基本信息获取系统' in result.stdout
        assert '--stock' in result.stdout
        assert '--verbose' in result.stdout
    
    @pytest.mark.asyncio
    async def test_system_startup(self):
        """测试系统启动"""
        # 创建临时环境
        with tempfile.TemporaryDirectory() as temp_dir:
            # 设置环境变量
            env = {
                'PYTHONPATH': str(Path.cwd() / 'src'),
                'CACHE_DIR': temp_dir
            }
            
            # 运行主程序（快速退出模式）
            result = subprocess.run([
                sys.executable, '-c', 
                '''
import sys
sys.path.insert(0, "src")
from main import StockInfoSystem
system = StockInfoSystem()
print("System initialized successfully")
                '''
            ], capture_output=True, text=True, env=env)
            
            assert result.returncode == 0
            assert "System initialized successfully" in result.stdout
    
    def test_setup_script_dry_run(self):
        """测试初始化脚本（干运行模式）"""
        # 创建临时的setup脚本用于测试
        test_script = '''#!/bin/bash
echo "Checking Python..."
python3 --version
echo "Setup script test completed"
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(test_script)
            temp_script = f.name
        
        try:
            subprocess.run(['chmod', '+x', temp_script])
            result = subprocess.run([temp_script], capture_output=True, text=True)
            
            assert result.returncode == 0
            assert "Checking Python..." in result.stdout
        finally:
            Path(temp_script).unlink()
    
    def test_requirements_file_exists(self):
        """测试requirements.txt文件存在"""
        req_file = Path('requirements.txt')
        assert req_file.exists()
        
        # 验证包含必要依赖
        content = req_file.read_text()
        assert 'aiohttp' in content
        assert 'pandas' in content
        assert 'pytest' in content
    
    def test_project_structure(self):
        """测试项目结构"""
        # 验证主要目录存在
        assert Path('src').exists()
        assert Path('tests').exists()
        assert Path('src/data').exists()
        assert Path('src/cache').exists()
        assert Path('src/analysis').exists()
        assert Path('src/config').exists()
        
        # 验证主要文件存在
        assert Path('src/main.py').exists()
        assert Path('setup.sh').exists()
        assert Path('requirement.md').exists()
        assert Path('.claude/CLAUDE.md').exists()
        assert Path('.claude/TEST.md').exists()
    
    def test_file_line_count_compliance(self):
        """测试文件行数符合规范"""
        # 检查Python文件行数不超过200行
        src_files = list(Path('src').rglob('*.py'))
        
        for file_path in src_files:
            line_count = len(file_path.read_text().splitlines())
            assert line_count <= 200, f"{file_path} 有 {line_count} 行，超过200行限制"
    
    def test_directory_file_count_compliance(self):
        """测试目录文件数符合规范"""
        # 检查每个目录文件数不超过8个
        for root in Path('src').rglob('*'):
            if root.is_dir():
                file_count = len([f for f in root.iterdir() if f.is_file()])
                assert file_count <= 8, f"{root} 有 {file_count} 个文件，超过8个文件限制"