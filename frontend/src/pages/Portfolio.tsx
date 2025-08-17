import React from 'react';
import { Card, Typography, Empty } from 'antd';
import { PieChartOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const Portfolio: React.FC = () => {
  return (
    <div style={{ paddingTop: 64 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <PieChartOutlined style={{ marginRight: 12 }} />
          投资组合
        </Title>
        <Text type="secondary">
          投资组合分析、优化和风险管理
        </Text>
      </div>

      <Card style={{ textAlign: 'center', minHeight: 400 }}>
        <Empty description="功能开发中" />
      </Card>
    </div>
  );
};

export default Portfolio;