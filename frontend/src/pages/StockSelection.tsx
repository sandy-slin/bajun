import React from 'react';
import { Card, Typography, Empty } from 'antd';
import { StockOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const StockSelection: React.FC = () => {
  return (
    <div style={{ paddingTop: 64 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <StockOutlined style={{ marginRight: 12 }} />
          股票筛选
        </Title>
        <Text type="secondary">
          智能股票筛选和个股分析
        </Text>
      </div>

      <Card style={{ textAlign: 'center', minHeight: 400 }}>
        <Empty description="功能开发中" />
      </Card>
    </div>
  );
};

export default StockSelection;