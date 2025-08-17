import React, { useState } from 'react';
import { Layout, Menu, Button } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  BarChartOutlined,
  StockOutlined,
  PieChartOutlined,
  RobotOutlined,
  SettingOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/dashboard',
      icon: <DashboardOutlined />,
      label: '控制台',
    },
    {
      key: '/sectors',
      icon: <BarChartOutlined />,
      label: '板块分析',
    },
    {
      key: '/stocks',
      icon: <StockOutlined />,
      label: '股票筛选',
    },
    {
      key: '/portfolio',
      icon: <PieChartOutlined />,
      label: '投资组合',
    },
    {
      key: '/trading',
      icon: <RobotOutlined />,
      label: '交易助手',
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: '系统设置',
    },
  ];

  const handleMenuClick = (key: string) => {
    navigate(key);
  };

  return (
    <Sider 
      trigger={null} 
      collapsible 
      collapsed={collapsed}
      style={{
        background: '#001529',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        zIndex: 100,
      }}
      width={240}
      collapsedWidth={80}
    >
      <div style={{
        height: 64,
        display: 'flex',
        alignItems: 'center',
        justifyContent: collapsed ? 'center' : 'space-between',
        padding: collapsed ? '0' : '0 16px',
        borderBottom: '1px solid #f0f0f0',
        background: '#002140',
      }}>
        {!collapsed && (
          <div style={{ 
            color: 'white', 
            fontWeight: 'bold', 
            fontSize: 18,
            display: 'flex',
            alignItems: 'center'
          }}>
            <RobotOutlined style={{ marginRight: 8, fontSize: 24, color: '#1890ff' }} />
            八骏平台
          </div>
        )}
        <Button
          type="text"
          icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
          onClick={() => setCollapsed(!collapsed)}
          style={{
            fontSize: '16px',
            width: 32,
            height: 32,
            color: 'white',
          }}
        />
      </div>
      
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        style={{ height: 'calc(100vh - 64px)', borderRight: 0 }}
        items={menuItems}
        onClick={({ key }) => handleMenuClick(key)}
      />
    </Sider>
  );
};

export default Sidebar;