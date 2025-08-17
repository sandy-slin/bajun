import React from 'react';
import { Card, Typography, Empty } from 'antd';
import { SettingOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

const Settings: React.FC = () => {
  return (
    <div style={{ paddingTop: 64 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2} style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <SettingOutlined style={{ marginRight: 12 }} />
          系统设置
        </Title>
        <Text type="secondary">
          系统配置和算法参数设置
        </Text>
      </div>

      <Card style={{ textAlign: 'center', minHeight: 400 }}>
        <Empty description="功能开发中" />
      </Card>
    </div>
  );
};

export default Settings;