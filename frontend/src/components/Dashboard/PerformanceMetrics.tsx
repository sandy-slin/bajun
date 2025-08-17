import React from 'react';
import { Card, Row, Col, Progress, Statistic, Tag, Typography, Divider } from 'antd';
import { 
  TrophyOutlined, 
  RiseOutlined, 
  CheckCircleOutlined,
  ExperimentOutlined 
} from '@ant-design/icons';
import { SystemStatus } from '../../services/websocket';

const { Title, Text } = Typography;

interface PerformanceMetricsProps {
  performanceData: any;
  systemStatus: SystemStatus | null;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ 
  performanceData, 
  systemStatus 
}) => {
  return (
    <Card 
      title={
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <TrophyOutlined style={{ marginRight: 8 }} />
          算法性能指标
        </div>
      }
    >
      <Row gutter={[24, 24]}>
        {/* 板块分析性能 */}
        <Col xs={24} lg={8}>
          <Card size="small" title="板块分析性能" style={{ height: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Progress
                type="circle"
                percent={69}
                strokeColor="#52c41a"
                format={() => '69.0%'}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>预测准确率</Text>
              </div>
            </div>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <div>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">基准准确率:</Text>
                <Text>64.0%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">当前准确率:</Text>
                <Text strong style={{ color: '#52c41a' }}>69.0%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">性能提升:</Text>
                <Tag color="green">+7.8%</Tag>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">稳定性评分:</Text>
                <Text>85/100</Text>
              </Row>
            </div>
          </Card>
        </Col>

        {/* 股票选择性能 */}
        <Col xs={24} lg={8}>
          <Card size="small" title="股票选择性能" style={{ height: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Progress
                type="circle"
                percent={50}
                strokeColor="#1890ff"
                format={() => '50.0%'}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>选股胜率</Text>
              </div>
            </div>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <div>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">基准胜率:</Text>
                <Text>40.0%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">当前胜率:</Text>
                <Text strong style={{ color: '#1890ff' }}>50.0%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">性能提升:</Text>
                <Tag color="blue">+25.0%</Tag>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">稳定性改善:</Text>
                <Text>+32.7%</Text>
              </Row>
            </div>
          </Card>
        </Col>

        {/* 投资组合性能 */}
        <Col xs={24} lg={8}>
          <Card size="small" title="投资组合性能" style={{ height: '100%' }}>
            <div style={{ textAlign: 'center', marginBottom: 16 }}>
              <Statistic
                value={0.31}
                precision={2}
                suffix="%"
                valueStyle={{ color: '#52c41a', fontSize: 24 }}
                prefix={<RiseOutlined />}
              />
              <div style={{ marginTop: 8 }}>
                <Text strong>组合收益率</Text>
              </div>
            </div>
            
            <Divider style={{ margin: '16px 0' }} />
            
            <div>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">基准收益:</Text>
                <Text style={{ color: '#cf1322' }}>-1.19%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">优化收益:</Text>
                <Text strong style={{ color: '#52c41a' }}>+0.31%</Text>
              </Row>
              <Row justify="space-between" style={{ marginBottom: 8 }}>
                <Text type="secondary">绝对改善:</Text>
                <Tag color="green">+1.50%</Tag>
              </Row>
              <Row justify="space-between">
                <Text type="secondary">相对改善:</Text>
                <Tag color="green">+126.1%</Tag>
              </Row>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 算法优化状态 */}
      {systemStatus?.optimization_status && (
        <div style={{ marginTop: 24 }}>
          <Title level={5}>
            <ExperimentOutlined style={{ marginRight: 8 }} />
            算法优化状态
          </Title>
          
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Card size="small" style={{ background: '#f6ffed', border: '1px solid #b7eb8f' }}>
                <div style={{ textAlign: 'center' }}>
                  <CheckCircleOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                  <div style={{ marginTop: 8 }}>
                    <Text strong>优化完成</Text>
                    <br />
                    <Text type="secondary">{systemStatus.optimization_status.improvement} 性能提升</Text>
                  </div>
                </div>
              </Card>
            </Col>
            
            <Col xs={24} sm={8}>
              <Card size="small">
                <Statistic
                  title="最后优化时间"
                  value={new Date(systemStatus.optimization_status.last_optimization).toLocaleString('zh-CN')}
                  valueStyle={{ fontSize: 14 }}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={8}>
              <Card size="small">
                <div style={{ textAlign: 'center' }}>
                  <Tag color="green" style={{ fontSize: 14, padding: '4px 12px' }}>
                    {systemStatus.optimization_status.status}
                  </Tag>
                  <div style={{ marginTop: 8 }}>
                    <Text type="secondary">系统状态</Text>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </div>
      )}
    </Card>
  );
};

export default PerformanceMetrics;