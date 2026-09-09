<template>
  <div class="app">
    <header class="header">
      <h1>AI 旅行规划师</h1>
      <p class="subtitle">RAG知识库 · 多Agent协作 · 高德地图 · 流式输出</p>
    </header>

    <main class="main">
      <div class="input-section">
        <div class="input-box">
          <textarea
            v-model="userInput"
            placeholder="描述你的旅行需求，例如：&#10;3天杭州游，预算2000，带老人小孩，不吃辣"
            rows="4"
            @keydown.enter.ctrl="submitPlan"
          ></textarea>
          <div class="input-options">
            <label class="checkbox-label">
              <input type="checkbox" v-model="useMemory" />
              <span>启用对话记忆</span>
            </label>
            <label class="checkbox-label">
              <input type="checkbox" v-model="useStream" />
              <span>流式输出</span>
            </label>
          </div>
          <button @click="submitPlan" :disabled="loading || !userInput.trim()">
            {{ loading ? '规划中...' : '开始规划' }}
          </button>
        </div>
        <div class="hints">
          <span @click="fillExample('3天杭州游，预算2000，带老人小孩，不吃辣')">杭州家庭游</span>
          <span @click="fillExample('5天北京深度游，预算5000，喜欢历史文化')">北京文化游</span>
          <span @click="fillExample('2天成都美食之旅，预算1500，吃货一枚')">成都美食游</span>
        </div>
      </div>

      <div v-if="threadId" class="thread-info">
        <span>会话ID: {{ threadId.substring(0, 8) }}...</span>
        <button @click="clearHistory" class="clear-btn">清除历史</button>
      </div>

      <div v-if="traces.length > 0" class="traces-section">
        <h3>执行过程</h3>
        <div class="traces">
          <div v-for="(trace, idx) in traces" :key="idx" class="trace-item">
            <span class="trace-step">{{ trace.step }}</span>
            <span class="trace-message">{{ trace.message }}</span>
          </div>
        </div>
      </div>

      <div v-if="streamingContent" class="result-section">
        <div class="result-card">
          <div class="plan-header">
            <span class="city">行程规划中...</span>
            <span class="status streaming">生成中</span>
          </div>
          <div class="plan-content streaming-content" v-html="renderedStreamContent"></div>
        </div>
      </div>

      <div v-else-if="result" class="result-section">
        <div class="result-card">
          <div class="plan-header">
            <span class="city">{{ result.plan?.city || '旅行规划' }}</span>
            <span class="days">{{ result.plan?.days }}天行程</span>
            <span class="status" :class="generationStatusClass">{{ generationStatusLabel }}</span>
          </div>
          <details v-if="result.sources?.length" class="rag-evidence" open>
            <summary>
              <span>RAG 参考证据（{{ result.sources.length }} 条）</span>
              <small>点击收起/展开</small>
            </summary>
            <div class="evidence-grid">
              <article v-for="source in result.sources" :key="`${source.kind}-${source.city}-${source.title || source.source}`" class="evidence-item">
                <div class="evidence-title">
                  <span class="evidence-kind">{{ sourceKindLabel(source.kind) }}</span>
                  <strong>{{ source.title || '未命名知识条目' }}</strong>
                  <span v-if="source.city" class="evidence-city">{{ source.city }}</span>
                </div>
                <p v-if="source.excerpt" class="evidence-excerpt">{{ source.excerpt }}</p>
                <div class="evidence-meta">
                  <span v-if="source.score > 0">相似度 {{ Number(source.score).toFixed(3) }}</span>
                  <span v-if="source.verified_at">核验日期 {{ source.verified_at }}</span>
                </div>
                <a
                  v-if="externalSourceUrl(source.url)"
                  :href="externalSourceUrl(source.url)"
                  target="_blank"
                  rel="noopener noreferrer"
                >打开外部原始来源 ↗</a>
                <span v-else class="local-source">本地知识库：{{ source.source }}</span>
              </article>
            </div>
          </details>
          <div v-else class="rag-evidence-empty">
            本次未命中目标城市的知识库证据，最终答案由 LLM 与实时工具结果生成。
          </div>
          <div class="plan-content" v-html="renderedMarkdown"></div>
        </div>
      </div>

      <div v-if="error" class="error-section">
        <p>{{ error }}</p>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

export default {
  name: 'App',
  data() {
    return {
      userInput: '',
      loading: false,
      result: null,
      traces: [],
      error: null,
      threadId: null,
      useMemory: true,
      useStream: true,
      streamingContent: '',
    }
  },
  computed: {
    renderedMarkdown() {
      if (!this.result?.answer) return ''
      return DOMPurify.sanitize(marked(this.result.answer))
    },
    renderedStreamContent() {
      if (!this.streamingContent) return ''
      return DOMPurify.sanitize(marked(this.streamingContent))
    },
    generationMeta() {
      return this.result?.generation_meta || {}
    },
    generationStatusLabel() {
      const mode = this.generationMeta.mode || 'fallback'
      const evidenceCount = this.result?.sources?.length || 0
      if (mode === 'llm_rag') return `RAG + LLM · ${evidenceCount}条证据`
      if (mode === 'llm_only') return '仅 LLM · 未命中知识库'
      if (mode === 'rag_only') return `仅 RAG · ${evidenceCount}条证据`
      return '降级结果'
    },
    generationStatusClass() {
      const mode = this.generationMeta.mode || 'fallback'
      return {
        streaming: false,
        success: mode === 'llm_rag',
        warning: mode === 'rag_only' || mode === 'llm_only',
        error: mode === 'fallback',
      }
    }
  },
  methods: {
    sourceKindLabel(kind) {
      if (kind === 'attraction') return '景点'
      if (kind === 'restaurant') return '餐厅'
      return '资料'
    },
    externalSourceUrl(url) {
      return /^https?:\/\//i.test(url || '') ? url : ''
    },
    fillExample(text) {
      this.userInput = text
    },
    async submitPlan() {
      if (!this.userInput.trim() || this.loading) return

      this.loading = true
      this.result = null
      this.traces = []
      this.error = null
      this.streamingContent = ''

      try {
        if (this.useStream) {
          await this.submitWithStream()
        } else {
          await this.submitNormal()
        }
      } catch (err) {
        this.error = err.response?.data?.detail || err.message || '规划失败，请稍后重试'
        console.error(err)
      } finally {
        this.loading = false
      }
    },
    async submitNormal() {
      const response = await axios.post('/api/plan', {
        question: this.userInput,
        thread_id: this.threadId,
        use_memory: this.useMemory,
      })

      this.result = response.data
      this.traces = response.data.traces || []
      this.threadId = response.data.thread_id
    },
    async submitWithStream() {
      const response = await fetch('/api/plan/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.userInput,
          thread_id: this.threadId,
          use_memory: this.useMemory,
        }),
      })

      if (!response.ok) {
        throw new Error(`请求失败（HTTP ${response.status}）`)
      }
      if (!response.body) {
        throw new Error('服务器没有返回流式响应')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          buffer += decoder.decode()
          if (buffer.trim()) this.processSseBlock(buffer)
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split(/\r?\n\r?\n/)
        buffer = blocks.pop() || ''

        for (const block of blocks) {
          this.processSseBlock(block)
        }
      }
    },
    processSseBlock(block) {
      const dataLine = block
        .split(/\r?\n/)
        .find((line) => line.startsWith('data: '))
      if (!dataLine) return

      const data = JSON.parse(dataLine.slice(6))
      this.handleStreamEvent(data)
    },
    handleStreamEvent(event) {
      switch (event.event) {
        case 'start':
          const startData = JSON.parse(event.data)
          this.threadId = startData.thread_id
          break
        case 'complete':
          const completeData = JSON.parse(event.data)
          this.result = completeData
          this.traces = completeData.traces || []
          this.streamingContent = ''
          break
        case 'progress':
          const progressData = JSON.parse(event.data)
          this.traces.push({ step: 'system', message: progressData.message })
          break
        case 'error':
          const errorData = JSON.parse(event.data)
          this.error = errorData.error
          throw new Error(this.error || '规划失败')
        case 'token':
          this.streamingContent += event.data
          break
      }
    },
    async clearHistory() {
      if (!this.threadId) return

      try {
        await axios.delete(`/api/history/${this.threadId}`)
        this.threadId = null
        this.result = null
        this.traces = []
      } catch (err) {
        console.error('Failed to clear history:', err)
      }
    }
  }
}
</script>

<style>
.app {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px;
}

.header {
  text-align: center;
  margin-bottom: 32px;
}

.header h1 {
  font-size: 32px;
  color: #1a1a2e;
  margin-bottom: 8px;
}

.subtitle {
  color: #666;
  font-size: 14px;
}

.input-section {
  margin-bottom: 24px;
}

.input-box {
  background: white;
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.input-box textarea {
  width: 100%;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 12px;
  font-size: 16px;
  resize: none;
  font-family: inherit;
}

.input-box textarea:focus {
  outline: none;
  border-color: #2563eb;
}

.input-options {
  display: flex;
  gap: 16px;
  margin-top: 12px;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  color: #666;
  cursor: pointer;
}

.checkbox-label input {
  cursor: pointer;
}

.input-box button {
  margin-top: 12px;
  width: 100%;
  padding: 12px;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.2s;
}

.input-box button:hover:not(:disabled) {
  background: #1d4ed8;
}

.input-box button:disabled {
  background: #94a3b8;
  cursor: not-allowed;
}

.hints {
  margin-top: 12px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.hints span {
  padding: 6px 12px;
  background: #e0e7ff;
  color: #3730a3;
  border-radius: 16px;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.2s;
}

.hints span:hover {
  background: #c7d2fe;
}

.thread-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: #f0f9ff;
  border-radius: 8px;
  font-size: 13px;
  color: #0369a1;
}

.clear-btn {
  padding: 4px 12px;
  background: #e0e7ff;
  color: #3730a3;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
}

.clear-btn:hover {
  background: #c7d2fe;
}

.traces-section {
  margin-bottom: 24px;
}

.traces-section h3 {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.traces {
  background: #f8fafc;
  border-radius: 8px;
  padding: 12px;
  max-height: 200px;
  overflow-y: auto;
}

.trace-item {
  display: flex;
  gap: 12px;
  padding: 6px 0;
  border-bottom: 1px solid #e2e8f0;
  font-size: 13px;
}

.trace-item:last-child {
  border-bottom: none;
}

.trace-step {
  color: #2563eb;
  font-weight: 500;
  min-width: 100px;
}

.trace-message {
  color: #64748b;
}

.result-section {
  margin-top: 24px;
}

.result-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.plan-header {
  background: linear-gradient(135deg, #2563eb, #7c3aed);
  color: white;
  padding: 20px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.plan-header .city {
  font-size: 24px;
  font-weight: 600;
}

.plan-header .days {
  background: rgba(255,255,255,0.2);
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 14px;
}

.plan-header .status {
  padding: 4px 12px;
  border-radius: 16px;
  font-size: 12px;
}

.plan-header .status.streaming {
  background: #22c55e;
  animation: pulse 1.5s infinite;
}

.plan-header .status.success {
  background: #2563eb;
  color: #fff;
}

.plan-header .status.warning {
  background: #f59e0b;
  color: #fff;
}

.plan-header .status.error {
  background: #9ca3af;
  color: #fff;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.plan-content {
  padding: 24px;
  line-height: 1.8;
}

.streaming-content {
  min-height: 200px;
}

.plan-content h2 {
  color: #1e40af;
  margin: 24px 0 12px;
  font-size: 20px;
  border-left: 4px solid #2563eb;
  padding-left: 12px;
}

.plan-content h3 {
  color: #374151;
  margin: 16px 0 8px;
  font-size: 16px;
}

.plan-content ul {
  margin: 8px 0;
  padding-left: 24px;
}

.plan-content li {
  margin: 4px 0;
  color: #4b5563;
}

.rag-evidence {
  margin: 20px 24px 0;
  border: 1px solid #bfdbfe;
  border-radius: 12px;
  background: #eff6ff;
  overflow: hidden;
}

.rag-evidence summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  color: #1e40af;
  font-weight: 700;
  cursor: pointer;
}

.rag-evidence summary small {
  color: #64748b;
  font-weight: 400;
}

.evidence-grid {
  display: grid;
  gap: 10px;
  padding: 0 14px 14px;
}

.evidence-item {
  padding: 12px;
  border: 1px solid #dbeafe;
  border-radius: 9px;
  background: #fff;
}

.evidence-title {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  color: #1f2937;
}

.evidence-kind {
  padding: 2px 8px;
  border-radius: 999px;
  color: #1d4ed8;
  background: #dbeafe;
  font-size: 12px;
}

.evidence-city,
.evidence-meta,
.local-source {
  color: #64748b;
  font-size: 12px;
}

.evidence-city::before {
  content: '· ';
}

.evidence-excerpt {
  margin: 8px 0;
  color: #475569;
  font-size: 13px;
  line-height: 1.55;
  white-space: pre-wrap;
}

.evidence-meta {
  display: flex;
  gap: 14px;
  margin-bottom: 5px;
}

.local-source {
  display: block;
  overflow-wrap: anywhere;
}

.rag-evidence-empty {
  margin: 20px 24px 0;
  padding: 12px 14px;
  border: 1px solid #fde68a;
  border-radius: 10px;
  color: #92400e;
  background: #fffbeb;
  font-size: 13px;
}

.error-section {
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 8px;
  padding: 16px;
  margin-top: 24px;
}

.error-section p {
  color: #dc2626;
}
</style>
