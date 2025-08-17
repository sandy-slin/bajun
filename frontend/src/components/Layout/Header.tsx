import React, { useEffect, useState } from 'react';
import { Layout, Space, Badge, Dropdown, Menu, Typography, Divider } from 'antd';
import {
  BellOutlined,
  WifiOutlined,
  DisconnectOutlined,
  UserOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons';
import { useWebSocket } from '../../contexts/WebSocketContext';
import { useApi } from '../../contexts/ApiContext';

const { Header: AntHeader } = Layout;
const { Text } = Typography;

const Header: React.FC = () => {
  const { isConnected, connectionState, systemStatus } = useWebSocket();
  const { healthCheck } = useApi();
  const [apiStatus, setApiStatus] = useState<'healthy' | 'unhealthy' | 'checking'>('checking');

  useEffect(() => {
    // 检查API健康状态
    const checkApiHealth = async () => {
      setApiStatus('checking');
      try {
        const result = await healthCheck();
        setApiStatus(result ? 'healthy' : 'unhealthy');
      } catch {
        setApiStatus('unhealthy');
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000); // 每30秒检查一次

    return () => clearInterval(interval);
  }, [healthCheck]);

  const getConnectionIcon = () => {
    return isConnected ? (
      <WifiOutlined style={{ color: '#52c41a' }} />
    ) : (
      <DisconnectOutlined style={{ color: '#ff4d4f' }} />
    );
  };

  const getApiStatusIcon = () => {
    switch (apiStatus) {
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'unhealthy':
        return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ApiOutlined style={{ color: '#faad14' }} />;
    }
  };

  const statusMenu = (
    <Menu>
      <Menu.Item key="connection">
        <Space>
          {getConnectionIcon()}
          <div>
            <Text strong>WebSocket连接</Text>
            <br />
            <Text type="secondary">状态: {isConnected ? '已连接' : '已断开'}</Text>
          </div>
        </Space>
      </Menu.Item>
      <Menu.Divider />
      <Menu.Item key="api">
        <Space>
          {getApiStatusIcon()}
          <div>
            <Text strong>API服务</Text>
            <br />
            <Text type="secondary">
              状态: {apiStatus === 'healthy' ? '正常' : apiStatus === 'unhealthy' ? '异常' : '检查中'}
            </Text>
          </div>
        </Space>
      </Menu.Item>
      {systemStatus && (
        <>
          <Menu.Divider />
          <Menu.Item key="performance">
            <div>
              <Text strong>系统性能</Text>
              <br />
              <Text type="secondary">板块准确率: {systemStatus.performance_metrics.sector_accuracy}</Text>
              <br />
              <Text type="secondary">股票胜率: {systemStatus.performance_metrics.stock_win_rate}</Text>
              <br />
              <Text type="secondary">组合收益: {systemStatus.performance_metrics.portfolio_return}</Text>
            </div>
          </Menu.Item>
        </>
      )}
    </Menu>
  );

  const userMenu = (
    <Menu>
      <Menu.Item key="profile">
        <UserOutlined /> 个人资料
      </Menu.Item>
      <Menu.Item key="settings">
        <UserOutlined /> 账户设置
      </Menu.Item>
      <Menu.Divider />
      <Menu.Item key="logout">
        <UserOutlined /> 退出登录
      </Menu.Item>
    </Menu>
  );

  return (
    <AntHeader 
      style={{ 
        padding: '0 24px',
        background: '#fff',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderBottom: '1px solid #f0f0f0',
        position: 'fixed',
        top: 0,
        right: 0,
        left: 0,
        zIndex: 99,
        height: 64,
        width: '100%'
      }}
    >
      <div>
        <Text strong style={{ fontSize: 16 }}>
          A股智能交易决策平台
        </Text>
        <Text type="secondary" style={{ marginLeft: 16 }}>
          {new Date().toLocaleDateString('zh-CN', {
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            weekday: 'long'
          })}
        </Text>
      </div>

      <Space size="large">
        {/* 系统状态指示器 */}
        <Dropdown overlay={statusMenu} placement="bottomRight">
          <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
            <Space>
              {getConnectionIcon()}
              {getApiStatusIcon()}
              <Text type="secondary">系统状态</Text>
            </Space>
          </div>
        </Dropdown>

        {/* 通知中心 */}
        <Badge count={0} size="small">
          <BellOutlined 
            style={{ 
              fontSize: 18,
              color: '#666',
              cursor: 'pointer',
            }} 
          />
        </Badge>

        {/* 用户菜单 */}
        <Dropdown overlay={userMenu} placement="bottomRight">
          <div style={{ 
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            padding: '4px 8px',
            borderRadius: '6px',
            background: '#f5f5f5'
          }}>
            <UserOutlined style={{ marginRight: 8 }} />
            <Text>用户</Text>
          </div>
        </Dropdown>
      </Space>
    </AntHeader>
  );
};

export default Header;