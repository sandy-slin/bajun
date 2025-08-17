import React from 'react';
import { Card, Typography, Empty } from 'antd';
import { RobotOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const TradingAssistant: React.FC = () => {
  return (
    <div style={{ paddingTop: 64 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <RobotOutlined style={{ marginRight: 12 }} />
          交易助手
        </Title>
        <Text type="secondary">
          反人性交易决策检查和情绪控制
        </Text>
      </div>

      <Card style={{ textAlign: 'center', minHeight: 400 }}>
        <Empty description="功能开发中" />
      </Card>
    </div>
  );
};

export default TradingAssistant;