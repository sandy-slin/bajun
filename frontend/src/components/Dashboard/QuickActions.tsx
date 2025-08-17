import React from 'react';
import { Card, Button, Space, Typography, List, Badge } from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  BarChartOutlined,
  StockOutlined,
  PieChartOutlined,
  RobotOutlined,
  PlayCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons';

const { Title, Text } = Typography;

const QuickActions: React.FC = () => {
  const navigate = useNavigate();

  const quickActionItems = [
    {
      key: 'sector-analysis',
      title: '板块分析',
      description: '获取TOP板块投资机会',
      icon: <BarChartOutlined style={{ color: '#1890ff' }} />,
      action: () => navigate('/sectors'),
      badge: 'HOT'
    },
    {
      key: 'stock-selection',
      title: '股票筛选',
      description: '智能选股和个股分析',
      icon: <StockOutlined style={{ color: '#52c41a' }} />,
      action: () => navigate('/stocks'),
      badge: null
    },
    {
      key: 'portfolio-optimize',
      title: '组合优化',
      description: '投资组合分析与优化',
      icon: <PieChartOutlined style={{ color: '#faad14' }} />,
      action: () => navigate('/portfolio'),
      badge: null
    },
    {
      key: 'trading-assistant',
      title: '交易助手',
      description: '反人性交易决策检查',
      icon: <RobotOutlined style={{ color: '#722ed1' }} />,
      action: () => navigate('/trading'),
      badge: 'NEW'
    }
  ];

  const systemActions = [
    {
      key: 'run-analysis',
      title: '运行完整分析',
      description: '执行板块+选股+组合的完整流程',
      icon: <PlayCircleOutlined style={{ color: '#13c2c2' }} />,
      action: () => {
        // TODO: 实现完整分析流程
        console.log('运行完整分析');
      }
    },
    {
      key: 'system-settings',
      title: '系统设置',
      description: '调整算法参数和系统配置',
      icon: <SettingOutlined style={{ color: '#8c8c8c' }} />,
      action: () => navigate('/settings')
    }
  ];

  return (
    <Card title="快速操作">
      <div style={{ marginBottom: 24 }}>
        <Title level={5} style={{ marginBottom: 16 }}>
          分析工具
        </Title>
        <List
          dataSource={quickActionItems}
          renderItem={item => (
            <List.Item style={{ padding: '12px 0' }}>
              <div
                style={{ 
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  cursor: 'pointer',
                  padding: 12,
                  borderRadius: 6,
                  transition: 'background-color 0.3s',
                }}
                onClick={item.action}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#f5f5f5';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                <div style={{ marginRight: 12, fontSize: 20 }}>
                  {item.icon}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <Text strong style={{ marginRight: 8 }}>
                      {item.title}
                    </Text>
                    {item.badge && (
                      <Badge 
                        count={item.badge} 
                        style={{ 
                          backgroundColor: item.badge === 'HOT' ? '#ff4d4f' : '#52c41a',
                          fontSize: 10
                        }} 
                      />
                    )}
                  </div>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {item.description}
                  </Text>
                </div>
              </div>
            </List.Item>
          )}
        />
      </div>

      <div>
        <Title level={5} style={{ marginBottom: 16 }}>
          系统操作
        </Title>
        <Space direction="vertical" style={{ width: '100%' }}>
          {systemActions.map(action => (
            <Button
              key={action.key}
              block
              size="large"
              icon={action.icon}
              onClick={action.action}
              style={{ 
                height: 'auto',
                padding: '12px 16px',
                textAlign: 'left',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              <div style={{ marginLeft: 8 }}>
                <div style={{ fontWeight: 500 }}>
                  {action.title}
                </div>
                <div style={{ 
                  fontSize: 12, 
                  color: '#8c8c8c',
                  fontWeight: 'normal'
                }}>
                  {action.description}
                </div>
              </div>
            </Button>
          ))}
        </Space>
      </div>
    </Card>
  );
};

export default QuickActions;