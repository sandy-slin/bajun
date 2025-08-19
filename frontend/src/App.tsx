import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import Dashboard from './pages/Dashboard';
import SectorAnalysis from './pages/SectorAnalysis';
import SectorAnalysisEnhanced from './pages/SectorAnalysisEnhanced';
import StockSelection from './pages/StockSelection';
import StockRecommendation from './pages/StockRecommendation';
import Portfolio from './pages/Portfolio';
import TradingAssistant from './pages/TradingAssistant';
import Settings from './pages/Settings';
import { DataProvider } from './contexts/DataContext';
import { ApiProvider } from './contexts/ApiContext';
import './App.css';

const { Content } = Layout;

function App() {
  return (
    <ApiProvider>
      <DataProvider>
        <Router>
          <Layout style={{ minHeight: '100vh' }}>
            <Sidebar />
            <Layout style={{ marginLeft: 240 }}>
              <Header />
              <Content style={{ 
                padding: '24px', 
                background: '#f5f5f5',
                marginTop: 64, // 为固定的Header留出空间
                minHeight: 'calc(100vh - 64px)' // 确保内容区域有足够高度
              }}>
                <div className="fade-in">
                  <Routes>
                    <Route path="/" element={<Dashboard />} />
                    <Route path="/dashboard" element={<Dashboard />} />
                    <Route path="/sectors" element={<SectorAnalysisEnhanced />} />
                    <Route path="/sectors-basic" element={<SectorAnalysis />} />
                    <Route path="/stocks" element={<StockSelection />} />
                    <Route path="/stock-recommendation" element={<StockRecommendation />} />
                    <Route path="/portfolio" element={<Portfolio />} />
                    <Route path="/trading" element={<TradingAssistant />} />
                    <Route path="/settings" element={<Settings />} />
                  </Routes>
                </div>
              </Content>
            </Layout>
          </Layout>
        </Router>
      </DataProvider>
    </ApiProvider>
  );
}

export default App;