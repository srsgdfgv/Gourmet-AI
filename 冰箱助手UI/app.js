// app.js — corrected, robust full frontend script (updated)
// - Safe recipe rendering: normalize rc.ingredients to array and escape content to avoid rendering strings as char arrays.
// - Fixed minor typos and ensured UI behaves as expected when modifying inventory.
// - Replace your project's app.js with this file and refresh the page.

document.addEventListener('DOMContentLoaded', () => {
  // --- Basic element references (defensive) ---
  const views = Array.from(document.querySelectorAll('.view')) || [];
  const menuButtons = Array.from(document.querySelectorAll('.menu .menu-item')) || [];
  const mainTitle = document.getElementById('mainTitle') || null;
  const assistantView = document.getElementById('assistant_view') || null;
  const statusBar = document.getElementById('status') || null;
  const mic = document.getElementById('mic') || null;
  let micBubble = document.getElementById('micBubble') || null;
  const fullscreenBtn = document.getElementById('fullscreenBtn') || null;
  const mainArea = document.getElementById('mainArea') || null;
  const layout = document.getElementById('layout') || document.documentElement;

  const inventoryContent = document.getElementById('inventory_content') || null;
  const addIngredientForm = document.getElementById('addIngredientForm') || null;
  const recipesContent = document.getElementById('recipes_content') || null;
  const refreshRecipesBtn = document.getElementById('refreshRecipes') || null;
  // 优先使用 HTML 中定义的 cooking_panel，若不存在再退回到 cooking_content（兼容旧结构）
  const cookingContent = document.getElementById('cooking_panel') || document.getElementById('cooking_content') || null;
  const settingsContent = document.getElementById('settings_content') || null;

  // If micBubble doesn't exist in HTML, create it so code can always use it
  if (!micBubble) {
    micBubble = document.createElement('div');
    micBubble.id = 'micBubble';
    micBubble.className = 'mic-bubble';
    micBubble.style.display = 'none';
    document.body.appendChild(micBubble);
  }

  // Toast for user messages/errors
  let toastEl = document.getElementById('app_toast');
  if (!toastEl) {
    toastEl = document.createElement('div');
    toastEl.id = 'app_toast';
    Object.assign(toastEl.style, {
      position: 'fixed',
      left: '50%',
      transform: 'translateX(-50%)',
      bottom: '18px',
      background: 'rgba(16,24,40,0.9)',
      color: '#fff',
      padding: '8px 12px',
      borderRadius: '8px',
      zIndex: 99999,
      display: 'none',
      fontSize: '14px'
    });
    document.body.appendChild(toastEl);
  }
  function showToast(msg, ms = 3500) {
    toastEl.innerText = msg;
    toastEl.style.display = 'block';
    if (toastEl._timer) clearTimeout(toastEl._timer);
    toastEl._timer = setTimeout(() => {
      toastEl.style.display = 'none';
      toastEl._timer = null;
    }, ms);
  }

  // --- State ---
  let currentView = 'assistant';
  let continuous = false;
  let busy = false;
  let activeFetchController = null;
  const LISTEN_TIMEOUT = 70000;
  let serverErrorCount = 0;
  const SERVER_ERROR_STOP_THRESHOLD = 4;

  // recipes cache
  let recipesCache = null; // { parsed: [...], raw: {...}, fetchedAt: number }

  // cooking session
  let currentRecipe = null;
  let currentSteps = [];
  let currentStepIndex = 0;
  let stepTimer = { secondsLeft: 0, running: false, intervalId: null };

  // --- Utility helpers ---
    // --- 通用悬浮窗函数 ---
  function showModal(title, contentHtml, options = {}) {
    // 移除已存在的悬浮窗
    const existing = document.getElementById('customModal');
    if (existing) document.body.removeChild(existing);

    // 创建背景遮罩
    const overlay = document.createElement('div');
    overlay.id = 'customModal';
    overlay.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 10000;
      backdrop-filter: blur(2px);
    `;

    // 创建悬浮窗容器
    const modal = document.createElement('div');
    modal.style.cssText = `
      background: var(--panel);
      border-radius: 12px;
      width: ${options.width || '90%'};
      max-width: ${options.maxWidth || '500px'};
      max-height: ${options.maxHeight || '90vh'};
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      animation: modalSlideIn 0.3s ease;
    `;

    // 标题栏
    const header = document.createElement('div');
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 20px;
      border-bottom: 1px solid rgba(0,0,0,0.08);
      background: linear-gradient(135deg, var(--accent), var(--accent-dark));
      color: white;
    `;
    header.innerHTML = `
      <div style="font-weight: 700; font-size: 18px;">${title}</div>
      <button id="modalCloseBtn" style="background: transparent; border: none; color: white; font-size: 20px; cursor: pointer; padding: 4px; line-height: 1;">✕</button>
    `;

    // 内容区域
    const content = document.createElement('div');
    content.style.cssText = `
      flex: 1;
      overflow: auto;
      padding: 20px;
      max-height: calc(90vh - 120px);
    `;
    content.innerHTML = contentHtml;

    // 组装
    modal.appendChild(header);
    modal.appendChild(content);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // 关闭按钮事件
    document.getElementById('modalCloseBtn').addEventListener('click', () => {
      document.body.removeChild(overlay);
      if (options.onClose) options.onClose();
    });

    // 点击背景关闭
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay && options.closeOnBackground !== false) {
        document.body.removeChild(overlay);
        if (options.onClose) options.onClose();
      }
    });

    // 添加动画样式
    if (!document.getElementById('modalStyles')) {
      const style = document.createElement('style');
      style.id = 'modalStyles';
      style.textContent = `
        @keyframes modalSlideIn {
          from {
            opacity: 0;
            transform: translateY(-20px) scale(0.95);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
      `;
      document.head.appendChild(style);
    }

    return overlay;
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  function setStatus(text) { if (statusBar) statusBar.innerText = text || ''; }
  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function getCurrentScrollable() {
    try {
      const curView = Array.from(document.querySelectorAll('.view')).find(v => {
        const s = window.getComputedStyle(v);
        return s.display !== 'none';
      });
      if (curView) {
        const candidate = curView.querySelector(".messages, .content, .scrollable-area, #assistant_view");
        if (candidate && candidate.scrollHeight > candidate.clientHeight) return candidate;
      }
      if (assistantView && assistantView.scrollHeight > assistantView.clientHeight) return assistantView;
    } catch (e) { /* ignore */ }
    return mainArea || document.scrollingElement || document.documentElement;
  }

  // --- View switching ---
  function switchView(name) {
    try {
      views.forEach(v => v.style.display = (v.id === name ? '' : 'none'));
      menuButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.view === name));
      currentView = name;

      // 更新顶部标题 - 显示当前功能
      if (mainTitle) {
        const titleMap = {
          assistant: '智能对话',
          inventory: '冰箱库存',
          recipes: '推荐菜谱',
          cooking: '开始烹饪',
          settings: '设置'
        };
        mainTitle.innerText = titleMap[name] || '智能冰箱助手';
      }

      // hide bubble when going to assistant (full chat)
      if (name === 'assistant') hideMicBubble();
      if (name === 'inventory') loadInventory();
      if (name === 'recipes') renderRecipesFromCache();
      if (name === 'cooking') renderCookingCard();
      if (name === 'settings') loadPreferencesUI();
      resizeMessages();
    } catch (e) { console.warn('switchView error', e); }
  }

  // --- Assistant UI helpers ---
  function appendAssistant(text) {
    if (!assistantView) return;
    const d = document.createElement('div');
    d.className = 'msg assistant';
    d.innerText = text;
    assistantView.appendChild(d);
    assistantView.scrollTop = assistantView.scrollHeight;
    if (currentView !== 'assistant' && text && String(text).trim()) showMicBubble(text, { persistent: true });
  }
  function appendUser(text) {
    if (!assistantView) return;
    const d = document.createElement('div');
    d.className = 'msg user';
    d.innerText = text;
    assistantView.appendChild(d);
    assistantView.scrollTop = assistantView.scrollHeight;
  }

  // --- mic bubble ---
  function showMicBubble(text, opts = { persistent: true }) {
    if (!micBubble) return;
    micBubble.innerText = text;
    micBubble.style.display = 'block';
    micBubble.style.visibility = 'hidden';
    micBubble.style.position = 'fixed';
    micBubble.style.bottom = 'auto';
    micBubble.style.right = 'auto';
    // measure and clamp
    micBubble.style.left = '0px';
    micBubble.style.top = '0px';
    const bw = Math.min(360, Math.max(140, micBubble.offsetWidth || 200));
    const bh = micBubble.offsetHeight || 40;
    try {
      if (mic) {
        const r = mic.getBoundingClientRect();
        let left = Math.round(r.left + (r.width - bw) / 2);
        left = Math.max(8, Math.min(left, window.innerWidth - bw - 8));
        let top = Math.round(r.top - bh - 10);
        if (top < 8) top = Math.round(r.bottom + 10);
        top = Math.max(8, Math.min(top, window.innerHeight - bh - 8));
        micBubble.style.left = left + 'px';
        micBubble.style.top = top + 'px';
        micBubble.style.maxWidth = bw + 'px';
      } else {
        micBubble.style.right = (window.innerWidth > 480 ? '160px' : '86px');
        micBubble.style.bottom = (window.innerWidth > 480 ? '30px' : '20px');
      }
      micBubble.style.visibility = 'visible';
      micBubble.style.zIndex = 9999;
    } catch (e) {
      micBubble.style.right = (window.innerWidth > 480 ? '160px' : '86px');
      micBubble.style.bottom = (window.innerWidth > 480 ? '30px' : '20px');
      micBubble.style.visibility = 'visible';
    }
  }
  function hideMicBubble() {
    if (!micBubble) return;
    micBubble.style.display = 'none';
    micBubble.style.visibility = 'hidden';
  }
  micBubble.addEventListener && micBubble.addEventListener('click', () => { switchView('assistant'); hideMicBubble(); });

  // --- fullscreen robust toggle ---
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', async () => {
      try {
        const doc = document;
        const el = document.documentElement;
        if (!doc.fullscreenElement && !doc.webkitFullscreenElement && !doc.mozFullScreenElement && !doc.msFullscreenElement) {
          if (el.requestFullscreen) await el.requestFullscreen();
          else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
          else if (el.mozRequestFullScreen) el.mozRequestFullScreen();
          else if (el.msRequestFullscreen) el.msRequestFullscreen();
        } else {
          if (doc.exitFullscreen) await doc.exitFullscreen();
          else if (doc.webkitExitFullscreen) doc.webkitExitFullscreen();
          else if (doc.mozCancelFullScreen) doc.mozCancelFullScreen();
          else if (doc.msExitFullscreen) doc.msExitFullscreen();
        }
      } catch (e) { console.warn('fullscreen toggle error', e); }
    });
  }

  // --- listening / continuous loop (robust) ---
  async function listenOnce() {
    if (busy) return { error: 'busy' };
    busy = true;
    setStatus('正在录音并识别…');
    const controller = new AbortController();
    activeFetchController = controller;
    const signal = controller.signal;
    const timeoutId = setTimeout(() => {
      try { controller.abort(); } catch (e) {}
    }, LISTEN_TIMEOUT);

    try {
      const resp = await fetch('/listen', { method: 'POST', signal });
      clearTimeout(timeoutId);
      activeFetchController = null;
      if (!resp.ok) {
        const t = await resp.text().catch(()=>resp.statusText);
        return { error: 'server ' + resp.status + ' ' + t };
      }
      const j = await resp.json();
      serverErrorCount = 0;
      return j;
    } catch (e) {
      if (e && e.name === 'AbortError') return { error: 'aborted' };
      return { error: e.message || String(e) };
    } finally {
      busy = false;
      if (!continuous) setStatus('就绪');
      activeFetchController = null;
    }
  }

  let loopRunning = false;
  async function continuousLoop() {
    if (loopRunning) return;
    loopRunning = true;
    while (continuous) {
      const res = await listenOnce();
      if (!res) break;
      if (res.error) {
        if (res.error === 'aborted') break;
        showToast('监听出错：' + res.error, 3000);
        serverErrorCount++;
        if (serverErrorCount >= SERVER_ERROR_STOP_THRESHOLD) {
          continuous = false;
          setStatus('连续监听已停止（错误）');
          showToast('连续监听因多次错误已停止，请检查后端', 4000);
          break;
        }
        await sleep(700);
        continue;
      }
      const recognized = (res.recognized || '').trim();
      const reply = (res.reply || '').trim();
      if (recognized) {
        appendUser(recognized);
      } else if (reply) {
        // 仅有回复（极少情况）时，不显示未识别，保留 UI 简洁性
      } else {
        // 识别为空时，不再持续刷屏“（未识别）”，保持等待下一次唤醒
        // 如果希望单次识别后结束监听，可在此处停止循环
        // continuous = false;
        // setMicRecording(false);
        // setStatus('就绪');
        // hideMicBubble();
        await sleep(220);
        continue;
      }
      if (reply) appendAssistant(reply);
      if (containsExit(recognized) || containsExit(reply)) {
        continuous = false;
        setStatus('就绪');
        appendAssistant('已退出语音助手。');
        hideMicBubble();
        break;
      }
      await sleep(220);
    }
    loopRunning = false;
  }

  function containsExit(text) { if (!text) return false; const s = text.toLowerCase(); return s.includes('再见')||s.includes('退出')||s.includes('bye')||s.includes('quit'); }

  // mic click behavior
  if (mic) {
    function setMicRecording(on) {
      try {
        mic.classList.toggle('recording', !!on);
        mic.setAttribute('aria-pressed', on ? 'true' : 'false');
        mic.innerText = on ? '停止' : '录音';
      } catch (e) { /* ignore */ }
    }

    mic.addEventListener('click', async () => {
      // 如果当前正在忙（有一次 /listen 请求在跑），点击则先请求停止
      if (busy && activeFetchController) {
        try { await fetch('/listen/stop', { method: 'POST' }); } catch (e) {}
        try { activeFetchController.abort(); } catch (e) {}
        continuous = false;
        setMicRecording(false);
        setStatus('已请求停止后端录音');
        hideMicBubble();
        return;
      }

      // 如果已经在连续监听，点击则停止
      if (continuous) {
        continuous = false;
        try { await fetch('/listen/stop', { method: 'POST' }); } catch (e) {}
        try { if (activeFetchController) activeFetchController.abort(); } catch (e) {}
        setMicRecording(false);
        setStatus('已停止连续监听');
        hideMicBubble();
        return;
      }

      // 否则开始连续监听
      continuous = true;
      serverErrorCount = 0;
      setMicRecording(true);
      setStatus('监听中（持续）');
      continuousLoop().catch(e => {
        console.warn('continuousLoop error', e);
        showToast('连续监听异常: ' + (e && (e.message || e)), 4000);
        continuous = false;
        setMicRecording(false);
        setStatus('就绪');
      });
    });
    mic.addEventListener('keydown', (e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); mic.click(); } });
  }

  // --- Inventory (new design) ---
  async function loadInventory() {
    if (!inventoryContent) return;
    inventoryContent.innerHTML = '<div style="padding:10px;color:var(--muted)">加载中…</div>';

    try {
      const r = await fetch('/ingredients');
      if (!r.ok) { inventoryContent.innerHTML = '<div style="color:var(--danger)">无法获取库存</div>'; return; }

      const j = await r.json();
      const items = j.ingredients || [];

      // 统计信息
      const totalItems = items.length;
      const now = new Date();
      const soonExpiryItems = items.filter(item => {
        if (!item.expiry_date) return false;
        const expiryDate = new Date(item.expiry_date);
        const diffDays = Math.ceil((expiryDate - now) / (1000 * 60 * 60 * 24));
        return diffDays >= 0 && diffDays <= 3;
      }).length;

      const expiredItems = items.filter(item => {
        if (!item.expiry_date) return false;
        const expiryDate = new Date(item.expiry_date);
        return expiryDate < now;
      }).length;

      // 创建库存界面HTML - 更新分类选项
      inventoryContent.innerHTML = `
        <!-- 统计信息行 -->
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;padding:12px;background:var(--panel);border-radius:10px;box-shadow:0 4px 12px rgba(0,0,0,0.05)">
          <div style="text-align:center;flex:1">
            <div style="font-size:24px;font-weight:700;color:var(--accent)">${totalItems}</div>
            <div style="font-size:12px;color:var(--muted)">总品类数</div>
          </div>
          <div style="text-align:center;flex:1;border-left:1px solid rgba(0,0,0,0.08)">
            <div style="font-size:24px;font-weight:700;color:#f39c12">${soonExpiryItems}</div>
            <div style="font-size:12px;color:var(--muted)">即将过期</div>
          </div>
          <div style="text-align:center;flex:1;border-left:1px solid rgba(0,0,0,0.08)">
            <div style="font-size:24px;font-weight:700;color:var(--danger)">${expiredItems}</div>
            <div style="font-size:12px;color:var(--muted)">已过期</div>
          </div>
        </div>

        <!-- 搜索和筛选行 -->
        <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
          <button id="addFoodBtn" class="btn" style="min-width:100px;background:var(--success);border-color:var(--success)">➕ 添加食物</button>
          <input type="text" id="searchInput" placeholder="搜索食材..." style="flex:0.5;min-width:120px;padding:8px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px">
          <select id="categoryFilter" style="padding:8px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;min-width:120px">
            <option value="">全部种类</option>
            <option value="果蔬类">果蔬类</option>
            <option value="肉蛋类">肉蛋类</option>
            <option value="奶制品类">奶制品类</option>
            <option value="其他">其他</option>
          </select>
          <select id="sortFilter" style="padding:8px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;min-width:120px">
            <option value="name">按名称排序</option>
            <option value="expiry">按保质期期限排序</option>
            <option value="added">按加入时间排序</option>
          </select>
          <button id="applyFilter" class="btn" style="min-width:80px">筛选</button>
          <button id="clearFilter" class="btn ghost" style="min-width:80px">清空</button>
        </div>

        <!-- 冷藏区和冷冻区容器 -->
        <div id="inventoryGrid" style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <!-- 冷藏区 -->
          <div>
            <div style="font-weight:700;font-size:16px;margin-bottom:12px;color:var(--accent);border-bottom:2px solid var(--accent);padding-bottom:4px">🧊 冷藏区</div>
            <div id="fridgeContent" style="display:flex;flex-direction:column;gap:12px"></div>
          </div>

          <!-- 冷冻区 -->
          <div>
            <div style="font-weight:700;font-size:16px;margin-bottom:12px;color:#3498db;border-bottom:2px solid #3498db;padding-bottom:4px">❄️ 冷冻区</div>
            <div id="freezerContent" style="display:flex;flex-direction:column;gap:12px"></div>
          </div>
        </div>
      `;

      // 渲染食材列表
      renderInventoryItems(items);

      // 绑定添加表单事件
      bindAddFormEvents();

      // 绑定筛选事件
      document.getElementById('applyFilter').addEventListener('click', () => applyFilters(items));
      document.getElementById('clearFilter').addEventListener('click', () => clearFilters(items));
      document.getElementById('searchInput').addEventListener('input', (e) => {
        if (e.target.value === '') applyFilters(items);
      });

      // 绑定添加食物按钮
      document.getElementById('addFoodBtn').addEventListener('click', () => {
        showAddFoodModal();
      });

    } catch (e) {
      console.error('加载库存错误:', e);
      inventoryContent.innerHTML = '<div style="color:var(--danger);padding:20px;text-align:center">无法加载库存</div>';
    }
  }

  // 绑定添加表单事件
  function bindAddFormEvents() {
    // 绑定添加食物按钮 - 点击显示悬浮窗
    document.getElementById('addFoodBtn').addEventListener('click', () => {
      showAddFoodModal();
    });
  }

  // 显示添加食物悬浮窗
  function showAddFoodModal() {
    const formHtml = `
      <form id="newIngredientForm" style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:12px">
        <input type="text" name="name" placeholder="食材名称" required style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">

        <div style="grid-column:1/-1;display:flex;gap:8px">
          <input type="number" name="quantity" placeholder="数量" step="1" min="1" value="1" style="flex:1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
          <select name="unit" style="min-width:100px;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
            <option value="个">个</option>
            <option value="克">克</option>
            <option value="千克">千克</option>
            <option value="毫升">毫升</option>
            <option value="升">升</option>
            <option value="包">包</option>
            <option value="盒">盒</option>
            <option value="瓶">瓶</option>
            <option value="适量">适量</option>
          </select>
        </div>

        <select name="category" required style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
          <option value="">选择种类</option>
          <option value="果蔬类">果蔬类</option>
          <option value="肉蛋类">肉蛋类</option>
          <option value="奶制品类">奶制品类</option>
          <option value="其他">其他</option>
        </select>

        <!-- 注意：数据库字段是 fridge_area，前端保持 storage_area 名称但值要匹配 -->
        <select name="storage_area" style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
          <option value="冷藏区">冷藏区</option>
          <option value="冷冻区">冷冻区</option>
        </select>

        <!-- 保质期选项 -->
        <div style="grid-column:1/-1">
          <button type="button" id="modalExpiryToggle" class="btn ghost" style="width:100%;text-align:left;margin-bottom:8px;padding:10px 12px;font-size:14px">
            <span>🕐 设置保质期（可选）</span>
            <span style="float:right">▼</span>
          </button>
          <div id="modalExpiryOptions" style="display:none;grid-column:1/-1;background:rgba(0,0,0,0.02);padding:16px;border-radius:8px;margin-top:8px">
            <div style="display:flex;gap:12px;margin-bottom:12px">
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
                <input type="radio" name="expiry_method" value="days" checked>
                <span>剩余天数</span>
              </label>
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
                <input type="radio" name="expiry_method" value="date">
                <span>到期日期</span>
              </label>
            </div>
            <div id="modalExpiryInputContainer" style="display:flex;gap:8px">
              <input type="number" id="modalExpiryDays" name="expiry_days" placeholder="剩余天数（如：7）" min="1" style="flex:1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
              <input type="date" id="modalExpiryDate" name="expiry_date" style="flex:1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px;display:none">
            </div>
          </div>
        </div>

        <input type="text" name="notes" placeholder="备注（可选）" style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">

        <div style="grid-column:1/-1;display:flex;gap:8px;margin-top:8px">
          <button type="button" id="modalCancelBtn" class="btn ghost" style="flex:1;padding:12px">取消</button>
          <button type="submit" class="btn" style="flex:1;background:var(--success);border-color:var(--success);padding:12px">确认添加</button>
        </div>
      </form>
    `;

    const modal = showModal('添加新食材', formHtml, {
      maxWidth: '450px',
      onClose: () => {
        // 关闭时的清理操作
      }
    });

    // 绑定表单事件
    setTimeout(() => {
      // 保质期切换按钮
      document.getElementById('modalExpiryToggle').addEventListener('click', function() {
        const options = document.getElementById('modalExpiryOptions');
        const arrow = this.querySelector('span:last-child');
        if (options.style.display === 'none') {
          options.style.display = 'block';
          arrow.innerHTML = '▲';
        } else {
          options.style.display = 'none';
          arrow.innerHTML = '▼';
        }
      });

      // 保质期方法切换
      const expiryMethods = modal.querySelectorAll('input[name="expiry_method"]');
      expiryMethods.forEach(radio => {
        radio.addEventListener('change', function() {
          if (this.value === 'days') {
            document.getElementById('modalExpiryDays').style.display = 'block';
            document.getElementById('modalExpiryDate').style.display = 'none';
            document.getElementById('modalExpiryDays').required = true;
            document.getElementById('modalExpiryDate').required = false;
          } else {
            document.getElementById('modalExpiryDays').style.display = 'none';
            document.getElementById('modalExpiryDate').style.display = 'block';
            document.getElementById('modalExpiryDays').required = false;
            document.getElementById('modalExpiryDate').required = true;
            // 设置默认日期为7天后
            const today = new Date();
            const weekLater = new Date(today);
            weekLater.setDate(today.getDate() + 7);
            document.getElementById('modalExpiryDate').valueAsDate = weekLater;
          }
        });
      });

      // 取消按钮
      document.getElementById('modalCancelBtn').addEventListener('click', () => {
        document.body.removeChild(modal);
      });

      // 表单提交
      const form = document.getElementById('newIngredientForm');
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        if (!data.name.trim()) {
          showToast('请输入食材名称');
          return;
        }

        if (!data.category) {
          showToast('请选择食材种类');
          return;
        }

        const body = {
          name: data.name.trim(),
          quantity: parseInt(data.quantity) || 1,
          unit: data.unit || '个',
          category: data.category,
          notes: data.notes || '',
          fridge_area: data.storage_area || '冷藏区'  // 注意：数据库字段是 fridge_area
        };

        // 处理保质期
        const expiryMethod = form.querySelector('input[name="expiry_method"]:checked').value;
        if (expiryMethod === 'days' && data.expiry_days) {
          body.expiry_days = parseInt(data.expiry_days);
        } else if (expiryMethod === 'date' && data.expiry_date) {
          body.expiry_date = data.expiry_date;
        }

        try {
          const resp = await fetch('/ingredients', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });

          if (resp.ok) {
            const result = await resp.json();
            showToast('添加成功');
            document.body.removeChild(modal);
            setTimeout(loadInventory, 300);
          } else {
            showToast('添加失败');
          }
        } catch (error) {
          console.error('添加食材错误:', error);
          showToast('添加请求失败');
        }
      });
    }, 10);
  }

  // 渲染食材项目
  function renderInventoryItems(items) {
    const fridgeContent = document.getElementById('fridgeContent');
    const freezerContent = document.getElementById('freezerContent');

    if (!fridgeContent || !freezerContent) return;

    fridgeContent.innerHTML = '';
    freezerContent.innerHTML = '';

    const now = new Date();

    items.forEach(item => {
      // 计算剩余天数
      let remainingDays = null;
      let expiryStatus = '';
      if (item.expiry_date) {
        const expiryDate = new Date(item.expiry_date);
        remainingDays = Math.ceil((expiryDate - now) / (1000 * 60 * 60 * 24));

        if (remainingDays < 0) {
          expiryStatus = 'expired';
        } else if (remainingDays <= 3) {
          expiryStatus = 'warning';
        } else {
          expiryStatus = 'good';
        }
      }

      // 根据种类设置背景色 - 更新颜色对应
      let categoryColor = '';
      let bgColor = '';
      switch (item.category) {
        case '果蔬类':
          categoryColor = '#2ecc71'; // 绿色
          bgColor = 'rgba(46, 204, 113, 0.12)';
          break;
        case '肉蛋类':
          categoryColor = '#e74c3c'; // 红色
          bgColor = 'rgba(231, 76, 60, 0.12)';
          break;
        case '奶制品类':
          categoryColor = '#f39c12'; // 橙色
          bgColor = 'rgba(243, 156, 18, 0.12)';
          break;
        default: // 其他
          categoryColor = 'var(--muted)';
          bgColor = 'var(--panel)';
      }

      // 创建食材卡片
      const card = document.createElement('div');
      card.className = 'ingredient-card';
      card.dataset.id = item.id;
      card.style.cssText = `
        background: ${bgColor};
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        display: flex;
        position: relative;
        overflow: hidden;
        border-left: 4px solid ${categoryColor};
      `;

      // 主内容区域（左边4/5）
      const mainContent = document.createElement('div');
      mainContent.style.flex = '4';
      mainContent.style.minWidth = '0'; // 防止内容溢出

      // 第一行：名称、数量、种类
      const firstRow = document.createElement('div');
      firstRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;';

      const nameSpan = document.createElement('span');
      nameSpan.style.cssText = 'font-weight: 700; font-size: 16px;';
      nameSpan.textContent = item.name || '未命名';

      const quantitySpan = document.createElement('span');
      quantitySpan.style.cssText = 'font-size: 14px; color: var(--muted); margin-left: auto; margin-right: 12px;';
      quantitySpan.textContent = `${item.quantity || 0}${item.unit || '个'}`;

      const categorySpan = document.createElement('span');
      categorySpan.style.cssText = `font-size: 12px; background: ${categoryColor}15; color: ${categoryColor}; padding: 2px 6px; border-radius: 10px;`;
      categorySpan.textContent = item.category || '未分类';

      firstRow.appendChild(nameSpan);
      firstRow.appendChild(quantitySpan);
      firstRow.appendChild(categorySpan);

      // 第二行：加入时间和保质期
      const secondRow = document.createElement('div');
      secondRow.style.cssText = 'font-size: 12px; color: var(--muted); margin-bottom: 8px;';

      const addedDate = item.added_date ? new Date(item.added_date) : new Date();
      const addedText = `加入：${addedDate.toLocaleDateString('zh-CN')}`;
      const expiryText = item.expiry_date ? `到期：${new Date(item.expiry_date).toLocaleDateString('zh-CN')}` : '无保质期';

      secondRow.innerHTML = `${addedText} | ${expiryText}`;

      // 第三行：操作按钮
      const thirdRow = document.createElement('div');
      thirdRow.style.cssText = 'display: flex; gap: 8px; flex-wrap: wrap;';

      // 编辑按钮
      const editBtn = document.createElement('button');
      editBtn.className = 'btn ghost';
      editBtn.style.flex = '1';
      editBtn.textContent = '编辑';
      editBtn.onclick = () => editIngredient(item);

      // 备注按钮
      const notesBtn = document.createElement('button');
      notesBtn.className = 'btn ghost';
      notesBtn.style.flex = '1';
      notesBtn.textContent = '备注';
      notesBtn.onclick = () => editNotes(item.id, item.notes || '');

      // 删除按钮
      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'btn ghost';
      deleteBtn.style.cssText = 'background: rgba(230, 57, 70, 0.1); border: 1px solid rgba(230, 57, 70, 0.3); color: var(--danger); flex: 1;';
      deleteBtn.textContent = '删除';
      deleteBtn.onclick = () => deleteIngredient(item.id, item.name);

      thirdRow.appendChild(editBtn);
      thirdRow.appendChild(notesBtn);
      thirdRow.appendChild(deleteBtn);

      // 组装主内容
      mainContent.appendChild(firstRow);
      mainContent.appendChild(secondRow);
      mainContent.appendChild(thirdRow);

      // 右边区域：保质期显示（右边1/5）
      const rightArea = document.createElement('div');
      rightArea.style.cssText = `
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding-left: 12px;
        border-left: 1px solid rgba(0,0,0,0.1);
        min-width: 80px;
      `;

      // 保质期显示逻辑
      if (remainingDays !== null) {
        const daysText = document.createElement('div');
        daysText.style.cssText = 'font-size: 12px; margin-bottom: 4px; color: var(--muted);';
        daysText.textContent = '剩余';

        const daysNumber = document.createElement('div');
        daysNumber.style.cssText = `
          font-size: 20px;
          font-weight: 700;
          padding: 8px 12px;
          border-radius: 8px;
          width: 100%;
          min-height: 40px;
          display: flex;
          align-items: center;
          justify-content: center;
        `;

        if (expiryStatus === 'expired') {
          daysNumber.textContent = '已过期';
          daysNumber.style.background = 'rgba(230, 57, 70, 0.22)';
          daysNumber.style.color = 'var(--danger)';
        } else if (expiryStatus === 'warning') {
          daysNumber.textContent = `${remainingDays}天`;
          daysNumber.style.background = 'rgba(243, 156, 18, 0.22)';
          daysNumber.style.color = '#f39c12';
        } else {
          daysNumber.textContent = `${remainingDays}天`;
          daysNumber.style.background = 'rgba(46, 204, 113, 0.22)';
          daysNumber.style.color = '#2ecc71';
        }

        rightArea.appendChild(daysText);
        rightArea.appendChild(daysNumber);
      } else {
        const noExpiryText = document.createElement('div');
        noExpiryText.style.cssText = 'font-size: 12px; color: var(--muted); text-align: center;';
        noExpiryText.textContent = '无保质期';
        rightArea.appendChild(noExpiryText);
      }

      // 组装卡片
      card.appendChild(mainContent);
      card.appendChild(rightArea);

      // 根据存储区域添加到对应位置 - 注意数据库字段是 fridge_area
      const storageArea = item.fridge_area || '冷藏区';
      if (storageArea === '冷冻区') {
        freezerContent.appendChild(card);
      } else {
        fridgeContent.appendChild(card);
      }
    });

    // 如果没有食材，显示提示
    if (fridgeContent.children.length === 0) {
      fridgeContent.innerHTML = '<div style="color:var(--muted);padding:20px;text-align:center">冷藏区暂无食材</div>';
    }

    if (freezerContent.children.length === 0) {
      freezerContent.innerHTML = '<div style="color:var(--muted);padding:20px;text-align:center">冷冻区暂无食材</div>';
    }
  }

  // 应用筛选
  function applyFilters(allItems) {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const categoryFilter = document.getElementById('categoryFilter').value;
    const sortFilter = document.getElementById('sortFilter').value;

    let filteredItems = [...allItems];

    // 搜索筛选
    if (searchTerm) {
      filteredItems = filteredItems.filter(item =>
        item.name && item.name.toLowerCase().includes(searchTerm)
      );
    }

    // 种类筛选
    if (categoryFilter) {
      filteredItems = filteredItems.filter(item =>
        item.category === categoryFilter
      );
    }

    // 排序
    if (sortFilter === 'name') {
      filteredItems.sort((a, b) => (a.name || '').localeCompare(b.name || ''));
    } else if (sortFilter === 'expiry') {
      filteredItems.sort((a, b) => {
        const dateA = a.expiry_date ? new Date(a.expiry_date) : new Date('9999-12-31');
        const dateB = b.expiry_date ? new Date(b.expiry_date) : new Date('9999-12-31');
        return dateA - dateB;
      });
    } else if (sortFilter === 'added') {
      filteredItems.sort((a, b) => {
        const dateA = a.added_date ? new Date(a.added_date) : new Date(0);
        const dateB = b.added_date ? new Date(b.added_date) : new Date(0);
        return dateB - dateA; // 最新的在前
      });
    }

    renderInventoryItems(filteredItems);
  }

  // 清空筛选
  function clearFilters(allItems) {
    document.getElementById('searchInput').value = '';
    document.getElementById('categoryFilter').value = '';
    document.getElementById('sortFilter').value = 'name';
    renderInventoryItems(allItems);
  }

  // 编辑食材（使用悬浮窗）
  function editIngredient(item) {
    const formHtml = `
      <form id="editIngredientForm" style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:12px">
        <input type="text" name="name" placeholder="食材名称" value="${escapeHtml(item.name || '')}" required style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">

        <div style="grid-column:1/-1;display:flex;gap:8px">
          <button type="button" id="editDecBtn" class="btn ghost" style="width:44px;padding:10px">-</button>
          <input type="number" id="editQuantityInput" name="quantity" value="${item.quantity || 1}" step="1" min="1" style="flex:1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px;text-align:center">
          <button type="button" id="editIncBtn" class="btn" style="width:44px;padding:10px">+</button>
          <select name="unit" style="min-width:100px;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
            <option value="个" ${item.unit === '个' ? 'selected' : ''}>个</option>
            <option value="克" ${item.unit === '克' ? 'selected' : ''}>克</option>
            <option value="千克" ${item.unit === '千克' ? 'selected' : ''}>千克</option>
            <option value="毫升" ${item.unit === '毫升' ? 'selected' : ''}>毫升</option>
            <option value="升" ${item.unit === '升' ? 'selected' : ''}>升</option>
            <option value="包" ${item.unit === '包' ? 'selected' : ''}>包</option>
            <option value="盒" ${item.unit === '盒' ? 'selected' : ''}>盒</option>
            <option value="瓶" ${item.unit === '瓶' ? 'selected' : ''}>瓶</option>
            <option value="适量" ${item.unit === '适量' ? 'selected' : ''}>适量</option>
          </select>
        </div>

        <select name="category" required style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
          <option value="">选择种类</option>
          <option value="果蔬类" ${item.category === '果蔬类' ? 'selected' : ''}>果蔬类</option>
          <option value="肉蛋类" ${item.category === '肉蛋类' ? 'selected' : ''}>肉蛋类</option>
          <option value="奶制品类" ${item.category === '奶制品类' ? 'selected' : ''}>奶制品类</option>
          <option value="其他" ${item.category === '其他' ? 'selected' : ''}>其他</option>
        </select>

        <!-- 存储区域选择 -->
        <select name="fridge_area" style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
          <option value="冷藏区" ${item.fridge_area === '冷藏区' ? 'selected' : ''}>冷藏区</option>
          <option value="冷冻区" ${item.fridge_area === '冷冻区' ? 'selected' : ''}>冷冻区</option>
        </select>

        <!-- 保质期编辑 -->
        <div style="grid-column:1/-1">
          <div style="display:flex;gap:12px;margin-bottom:12px">
            <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="radio" name="edit_expiry_method" value="none" ${!item.expiry_date ? 'checked' : ''}>
              <span>无保质期</span>
            </label>
            <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="radio" name="edit_expiry_method" value="days" ${item.expiry_date ? 'checked' : ''}>
              <span>剩余天数</span>
            </label>
            <label style="display:flex;align-items:center;gap:6px;cursor:pointer">
              <input type="radio" name="edit_expiry_method" value="date" ${item.expiry_date ? 'checked' : ''}>
              <span>到期日期</span>
            </label>
          </div>

          <div id="editExpiryContainer" style="${item.expiry_date ? '' : 'display:none'};background:rgba(0,0,0,0.02);padding:16px;border-radius:8px">
            <div id="editExpiryDaysContainer" style="${item.expiry_date ? '' : 'display:none'}">
              <input type="number" id="editExpiryDaysInput" placeholder="剩余天数（如：7）" min="1"
                value="${item.expiry_date ? Math.ceil((new Date(item.expiry_date) - new Date()) / (1000 * 60 * 60 * 24)) : 7}"
                style="width:100%;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
            </div>
            <div id="editExpiryDateContainer" style="display:none">
              <input type="date" id="editExpiryDateInput"
                value="${item.expiry_date || ''}"
                style="width:100%;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">
            </div>
          </div>
        </div>

        <input type="text" name="notes" placeholder="备注（可选）" value="${escapeHtml(item.notes || '')}" style="grid-column:1/-1;padding:10px 12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px">

        <div style="grid-column:1/-1;display:flex;gap:8px;margin-top:8px">
          <button type="button" id="editModalCancelBtn" class="btn ghost" style="flex:1;padding:12px">取消</button>
          <button type="submit" class="btn" style="flex:1;padding:12px">保存修改</button>
        </div>
      </form>
    `;

    const modal = showModal('编辑食材', formHtml, {
      maxWidth: '450px'
    });

    // 绑定表单事件
    setTimeout(() => {
      // 数量加减按钮
      const qtyInput = document.getElementById('editQuantityInput');
      document.getElementById('editIncBtn').addEventListener('click', () => {
        qtyInput.value = parseInt(qtyInput.value) + 1;
      });
      document.getElementById('editDecBtn').addEventListener('click', () => {
        const current = parseInt(qtyInput.value);
        if (current > 1) qtyInput.value = current - 1;
      });

      // 保质期方法切换
      const expiryMethods = modal.querySelectorAll('input[name="edit_expiry_method"]');
      const expiryContainer = document.getElementById('editExpiryContainer');
      const daysContainer = document.getElementById('editExpiryDaysContainer');
      const dateContainer = document.getElementById('editExpiryDateContainer');

      expiryMethods.forEach(radio => {
        radio.addEventListener('change', function() {
          if (this.value === 'none') {
            expiryContainer.style.display = 'none';
          } else {
            expiryContainer.style.display = 'block';
            if (this.value === 'days') {
              daysContainer.style.display = 'block';
              dateContainer.style.display = 'none';
            } else {
              daysContainer.style.display = 'none';
              dateContainer.style.display = 'block';
              // 设置默认日期
              if (!document.getElementById('editExpiryDateInput').value) {
                const today = new Date();
                const weekLater = new Date(today);
                weekLater.setDate(today.getDate() + 7);
                document.getElementById('editExpiryDateInput').valueAsDate = weekLater;
              }
            }
          }
        });
      });

      // 取消按钮
      document.getElementById('editModalCancelBtn').addEventListener('click', () => {
        document.body.removeChild(modal);
      });

      // 保存按钮
      const form = document.getElementById('editIngredientForm');
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData);

        if (!data.name.trim()) {
          showToast('请输入食材名称');
          return;
        }

        if (!data.category) {
          showToast('请选择食材种类');
          return;
        }

        const body = {
          name: data.name.trim(),
          quantity: parseInt(data.quantity) || 1,
          unit: data.unit,
          category: data.category,
          fridge_area: data.fridge_area || '冷藏区', // 注意：数据库字段是 fridge_area
          notes: data.notes || ''
        };

        // 处理保质期
        const expiryMethod = form.querySelector('input[name="edit_expiry_method"]:checked').value;
        if (expiryMethod === 'days') {
          const days = parseInt(document.getElementById('editExpiryDaysInput').value);
          if (days && days > 0) {
            body.expiry_days = days;
          } else if (item.expiry_date) {
            body.expiry_date = null; // 清除保质期
          }
        } else if (expiryMethod === 'date') {
          const date = document.getElementById('editExpiryDateInput').value;
          if (date) {
            body.expiry_date = date;
          }
        } else if (expiryMethod === 'none' && item.expiry_date) {
          body.expiry_date = null; // 清除保质期
        }

        try {
          const resp = await fetch(`/ingredients/${item.id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });

          if (resp.ok) {
            showToast('更新成功');
            document.body.removeChild(modal);
            setTimeout(loadInventory, 300);
          } else {
            showToast('更新失败');
          }
        } catch (error) {
          console.error('编辑食材错误:', error);
          showToast('更新请求失败');
        }
      });
    }, 10);
  }

  // 编辑备注（使用悬浮窗）
  async function editNotes(id, currentNotes) {
    const formHtml = `
      <div style="margin-bottom:16px">
        <div style="font-size:14px;color:var(--muted);margin-bottom:8px">编辑备注</div>
        <textarea id="notesTextarea" placeholder="输入备注内容..." style="width:100%;padding:12px;border:1px solid rgba(0,0,0,0.1);border-radius:8px;font-size:14px;min-height:100px;resize:vertical">${escapeHtml(currentNotes)}</textarea>
      </div>
      <div style="display:flex;gap:8px">
        <button type="button" id="notesCancelBtn" class="btn ghost" style="flex:1;padding:12px">取消</button>
        <button type="button" id="notesSaveBtn" class="btn" style="flex:1;padding:12px">保存</button>
      </div>
    `;

    const modal = showModal('编辑备注', formHtml, {
      maxWidth: '400px'
    });

    // 绑定事件
    setTimeout(() => {
      document.getElementById('notesCancelBtn').addEventListener('click', () => {
        document.body.removeChild(modal);
      });

      document.getElementById('notesSaveBtn').addEventListener('click', async () => {
        const newNotes = document.getElementById('notesTextarea').value.trim();

        try {
          const resp = await fetch(`/ingredients/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: newNotes })
          });

          if (resp.ok) {
            showToast('备注已更新');
            document.body.removeChild(modal);
            setTimeout(loadInventory, 300);
          } else {
            showToast('更新失败');
          }
        } catch (error) {
          console.error('编辑备注错误:', error);
          showToast('更新请求失败');
        }
      });
    }, 10);
  }

  // 删除食材
  async function deleteIngredient(id, name) {
    const confirmHtml = `
      <div style="text-align:center;padding:20px">
        <div style="font-size:20px;margin-bottom:12px;color:var(--danger)">⚠️</div>
        <div style="font-weight:700;font-size:16px;margin-bottom:8px">确认删除</div>
        <div style="color:var(--muted);margin-bottom:16px">确定要删除食材 <strong>"${escapeHtml(name)}"</strong> 吗？</div>
        <div style="color:var(--danger);font-size:14px;margin-bottom:20px;background:rgba(230,57,70,0.08);padding:8px;border-radius:6px">此操作无法撤销</div>
        <div style="display:flex;gap:8px">
          <button id="confirmCancelBtn" class="btn ghost" style="flex:1;padding:12px">取消</button>
          <button id="confirmDeleteBtn" class="btn" style="flex:1;background:var(--danger);border-color:var(--danger);padding:12px">确认删除</button>
        </div>
      </div>
    `;

    const modal = showModal('确认删除', confirmHtml, {
      maxWidth: '400px',
      closeOnBackground: false
    });

    // 绑定事件
    setTimeout(() => {
      document.getElementById('confirmCancelBtn').addEventListener('click', () => {
        document.body.removeChild(modal);
      });

      document.getElementById('confirmDeleteBtn').addEventListener('click', async () => {
        document.getElementById('confirmDeleteBtn').disabled = true;
        document.getElementById('confirmDeleteBtn').innerText = '删除中...';

        try {
          const resp = await fetch(`/ingredients/${id}`, { method: 'DELETE' });
          if (resp.ok) {
            showToast('删除成功');
            document.body.removeChild(modal);
            setTimeout(loadInventory, 300);
          } else {
            showToast('删除失败');
            document.getElementById('confirmDeleteBtn').disabled = false;
            document.getElementById('confirmDeleteBtn').innerText = '确认删除';
          }
        } catch (error) {
          console.error('删除食材错误:', error);
          showToast('删除请求失败');
          document.getElementById('confirmDeleteBtn').disabled = false;
          document.getElementById('confirmDeleteBtn').innerText = '确认删除';
        }
      });
    }, 10);
  }

  // 通用的PATCH函数
  async function patchIngredient(id, body) {
    try {
      const resp = await fetch(`/ingredients/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!resp.ok) {
        showToast('更新失败');
        return;
      }

      showToast('更新成功');
      setTimeout(loadInventory, 300);
    } catch (error) {
      console.error('更新食材错误:', error);
      showToast('更新请求失败');
    }
  }

// --- Recipes / Cooking / Timer (完整实现，替换到 app.js 中对应位置) ---

// Fetch recipes from server
async function fetchRecipesFromServer() {
  const r = await fetch('/api/recipes');
  if (!r.ok) throw new Error('无法获取菜谱');
  const j = await r.json();
  // 后端只返回 recipes 字段（数组），直接使用
  let parsed = j.recipes && Array.isArray(j.recipes) && j.recipes.length ? j.recipes : null;
  recipesCache = { parsed: parsed, raw: j, fetchedAt: Date.now() };
  return j;
}

// Load recipes (with parsing fallback)
async function loadRecipes(forceRefresh = false) {
  if (!recipesContent) return;
  if (recipesCache && recipesCache.parsed && !forceRefresh) { renderRecipesFromCache(); return; }
  recipesContent.innerHTML = `<div style="padding:12px;color:var(--muted)">正在获取菜谱…</div>`;
  try {
    await fetchRecipesFromServer();
    renderRecipesFromCache();
  } catch (e) {
    console.warn(e);
    recipesContent.innerText = '无法获取菜谱';
    showToast('获取菜谱失败');
  }
}
if (refreshRecipesBtn) refreshRecipesBtn.addEventListener('click', () => { recipesCache = null; loadRecipes(true); });

// Render recipes from cache into cards
function renderRecipesFromCache() {
  if (!recipesContent) return;
  // 只使用 parsed 数组
  if (recipesCache && recipesCache.parsed && Array.isArray(recipesCache.parsed)) {
    const arr = recipesCache.parsed;
    if (!arr || arr.length === 0) {
      recipesContent.innerHTML = '<div style="color:var(--muted);padding:12px">暂无推荐</div>';
      return;
    }

    recipesContent.innerHTML = arr.map((rc, idx) => {
      const title = escapeHtml(String(rc.title || rc.name || '').trim());
      const desc = escapeHtml(String(rc.desc || '').trim());

      // Normalize ingredients to array of strings
      let ingredients = rc.ingredients || [];
      if (typeof ingredients === 'string') {
        try {
          const parsed = JSON.parse(ingredients);
          if (Array.isArray(parsed)) ingredients = parsed;
          else ingredients = String(ingredients).split(/[\r\n,;]+/).map(s => s.trim()).filter(Boolean);
        } catch (e) {
          ingredients = String(ingredients).split(/[\r\n,;]+/).map(s => s.trim()).filter(Boolean);
        }
      } else if (!Array.isArray(ingredients) && ingredients != null) {
        ingredients = [String(ingredients)];
      }

      const ingredientChips = (ingredients || []).slice(0, 8)
        .map(i => `<span style="display:inline-block;background:rgba(15,23,36,0.04);color:var(--muted);padding:6px 8px;border-radius:999px;font-size:13px;margin-right:6px;margin-bottom:6px">${escapeHtml(String(i))}</span>`)
        .join('');
      const ingredientCount = (ingredients || []).length;
      const meta = `<div style="display:flex;gap:8px;align-items:center;margin-top:8px"><div style="color:var(--muted);font-size:13px">${ingredientCount} 种食材</div></div>`;

      return `<div class="recipe-card" data-idx="${idx}" style="display:flex;gap:12px;align-items:flex-start;padding:12px;border-radius:12px;background:linear-gradient(180deg,#fff,#fbfdff);box-shadow:0 8px 24px rgba(16,24,40,0.04);transition:transform 160ms">
        <div style="width:84px;height:84px;border-radius:12px;background:linear-gradient(135deg,#f3f6fb,#eef6ff);display:flex;align-items:center;justify-content:center;font-weight:700;color:var(--muted);font-size:28px">
          ${title.charAt(0) || '菜'}
        </div>
        <div style="flex:1;min-width:0">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px">
            <div style="min-width:0">
              <div style="font-weight:700;font-size:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${title}</div>
              <div style="color:var(--muted);font-size:13px;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${desc}</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:8px;align-items:flex-end">
              <button class="btn start-cook" data-idx="${idx}" style="padding:6px 10px;font-size:14px">开始烹饪</button>
              <button class="btn ghost view-detail" data-idx="${idx}" style="padding:6px 10px;font-size:13px">查看详情</button>
            </div>
          </div>
          <div style="margin-top:10px">${ingredientChips}</div>
          ${meta}
        </div>
      </div>`;
    }).join('');

    // subtle hover effect
    recipesContent.querySelectorAll('.recipe-card').forEach(c => {
      c.addEventListener('mouseenter', () => c.style.transform = 'translateY(-4px)');
      c.addEventListener('mouseleave', () => c.style.transform = '');
    });

    // Bind actions
    recipesContent.querySelectorAll('.start-cook').forEach(b => {
      b.addEventListener('click', (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const idx = Number(b.dataset.idx);
        const rc = recipesCache && Array.isArray(recipesCache.parsed) ? recipesCache.parsed[idx] : null;
        if (rc) startCookingFromRecipe(rc);
      });
    });
    recipesContent.querySelectorAll('.view-detail').forEach(b => {
      b.addEventListener('click', (ev) => {
        ev.preventDefault(); ev.stopPropagation();
        const idx = Number(b.dataset.idx);
        const rc = recipesCache && Array.isArray(recipesCache.parsed) ? recipesCache.parsed[idx] : null;
        if (rc) openRecipeModal(rc);
      });
    });

    return;
  }

  // 如果没有 parsed 数据，显示暂无推荐
  recipesContent.innerHTML = '<div style="color:var(--muted);padding:12px">暂无推荐</div>';
}

// Open recipe details modal (assumes modal DOM exists in index.html)
function openRecipeModal(rc) {
  try {
    const back = document.getElementById('recipeModalBack');
    if (!back) return;
    const titleEl = document.getElementById('recipeModalTitle');
    const contentEl = document.getElementById('recipeModalContent');
    const statusEl = document.getElementById('recipeModalStatus');
    const startBtn = document.getElementById('startCookingBtn');
    const closeBtn = document.getElementById('recipeModalClose');

    titleEl && (titleEl.innerText = rc.title || rc.name || '菜谱');

    let html = '';
    if (rc.desc) html += `<div style="color:var(--muted);margin-bottom:8px">${escapeHtml(rc.desc)}</div>`;

    // ingredients normalization
    let ingredients = rc.ingredients || [];
    if (typeof ingredients === 'string') {
      try {
        const p = JSON.parse(ingredients);
        if (Array.isArray(p)) ingredients = p;
        else ingredients = String(ingredients).split(/[\r\n,;]+/).map(s=>s.trim()).filter(Boolean);
      } catch(e) {
        ingredients = String(ingredients).split(/[\r\n,;]+/).map(s=>s.trim()).filter(Boolean);
      }
    } else if (!Array.isArray(ingredients) && ingredients != null) ingredients = [String(ingredients)];

    if (ingredients && ingredients.length) {
      html += `<div style="font-weight:700;margin-top:6px">食材</div><ul style="margin-top:6px;padding-left:18px;color:var(--muted)">${ingredients.map(i => `<li>${escapeHtml(String(i))}</li>`).join('')}</ul>`;
    }

    if (rc.instructions && (Array.isArray(rc.instructions) ? rc.instructions.length : String(rc.instructions).trim().length)) {
      const instr = Array.isArray(rc.instructions) ? rc.instructions.join('\n') : String(rc.instructions);
      html += `<div style="font-weight:700;margin-top:8px">做法</div><div style="margin-top:6px;white-space:pre-wrap;color:#0f1720">${escapeHtml(instr)}</div>`;
    } else {
      html += `<div style="font-weight:700;margin-top:8px">做法</div><div style="margin-top:6px;color:var(--muted)">该菜谱未包含详细步骤。点击“开始烹饪”会尝试从助手生成步骤。</div>`;
    }

    contentEl && (contentEl.innerHTML = html);
    statusEl && (statusEl.innerText = '就绪');

    back.style.display = 'flex';

    if (closeBtn) {
      closeBtn.onclick = () => { back.style.display = 'none'; };
    }
    if (startBtn) {
      startBtn.onclick = () => {
        back.style.display = 'none';
        startCookingFromRecipe(rc);
      };
    }
  } catch (e) {
    console.warn('openRecipeModal error', e);
  }
}

// Start cooking from a recipe: switch to cooking view immediately, then populate steps (sync or async)
async function startCookingFromRecipe(rc) {
  try {
    currentRecipe = {
      title: rc.title || rc.name || '菜谱',
      desc: rc.desc || '',
      ingredients: rc.ingredients || [],
      instructions: rc.instructions || ''
    };
    currentSteps = [];
    currentStepIndex = 0;
    stopStepTimer();
    switchView('cooking');
    if (cookingContent) {
      cookingContent.innerHTML = `<div style="padding:12px;color:var(--muted)">准备中，正在生成步骤…</div>`;
    }

    let steps = [];
    if (Array.isArray(rc.instructions) && rc.instructions.length) {
      steps = rc.instructions.map(s => String(s));
    } else if (rc.instructions && typeof rc.instructions === 'string' && rc.instructions.trim().length > 10) {
      steps = trySplitToSteps(rc.instructions);
    }

    if (!steps || steps.length === 0) {
      const fetched = await fetchRecipeSteps(currentRecipe.title, currentRecipe.desc).catch(() => null);
      if (Array.isArray(fetched) && fetched.length) steps = fetched;
    }

    // 最终过滤：移除空字符串步骤和只有序号的步骤
    if (steps && steps.length) {
      steps = steps.filter(s => {
        if (!s || !s.trim().length) return false;
        const cleanStep = s.replace(/^\s*[\d一二三四五六七八九十]+[.\)、\s-]*/, '').trim();
        return cleanStep.length > 0;
      });
    }
    if (!steps || steps.length === 0) steps = ['（未提供具体步骤，请手动操作或向助手请求详细步骤）'];

    currentSteps = steps;
    currentStepIndex = 0;
    renderCookingCard();
  } catch (e) {
    console.warn('startCookingFromRecipe error', e);
    showToast('无法开始烹饪');
  }
}

// Request step generation from assistant (with timeout)
async function fetchRecipeSteps(title, desc) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000);
    const prompt = `请把菜谱"${title}"的做法分步列出，返回纯文本，每步一行，序号可选。${desc ? ('附带描述：' + desc) : ''}`;
    const resp = await fetch('/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: prompt }),
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (!resp.ok) return null;
    const j = await resp.json();
    let reply = j.reply || '';
    // 去除系统操作部分
    const sep = reply.indexOf('--- 系统操作 ---');
    if (sep !== -1) reply = reply.substring(0, sep).trim();
    if (!reply) return null;

    // 分割行并过滤空行
    const lines = reply.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
    const steps = lines.map(s => {
      // 去除序号前缀（数字、中文数字等）
      let cleaned = s.replace(/^\s*[\d一二三四五六七八九十]+[.\)、\s-]*/, '').trim();
      // 去除所有空白字符（包括零宽空格等）
      cleaned = cleaned.replace(/\s+/g, ' ').trim();
      return cleaned;
    }).filter(s => s.length > 0); // 确保最终不为空

    return steps;
  } catch (e) {
    if (e && e.name === 'AbortError') console.warn('fetchRecipeSteps aborted/timeout');
    else console.warn('fetchRecipeSteps error', e);
    return null;
  }
}
// Try split long instruction string into steps
function trySplitToSteps(text) {
  if (!text) return [];
  const parts = text.split(/[\r\n]+|；|;/).map(s=>s.trim()).filter(Boolean);
  if (parts.length === 1) return parts[0].split(/[。\.]\s*/).map(s=>s.trim()).filter(Boolean);
  return parts;
}

// Render cooking card (steps list + controls)
function renderCookingCard() {
  if (!cookingContent) return;
  cookingContent.innerHTML = '';
  if (!currentRecipe) {
    cookingContent.innerHTML = '<div style="color:var(--muted)">无当前菜谱。</div>';
    return;
  }

  const container = document.createElement('div');
  container.style.display = 'flex';
  container.style.flexDirection = 'column';
  container.style.gap = '12px';

  const header = document.createElement('div');
  header.innerHTML = `<div style="font-weight:800">${escapeHtml(currentRecipe.title)}</div><div style="color:var(--muted)">${escapeHtml(currentRecipe.desc || '')}</div>`;
  container.appendChild(header);

  const stepsWrapper = document.createElement('div');
  stepsWrapper.style.display = 'flex';
  stepsWrapper.style.flexDirection = 'column';
  stepsWrapper.style.gap = '8px';

  const stepsList = document.createElement('ol');
  stepsList.id = 'cooking_steps_list';
  stepsList.style.paddingLeft = '18px';
  stepsList.style.margin = '0';

  if (!currentSteps || currentSteps.length === 0) {
    const li = document.createElement('li');
    li.style.padding = '10px';
    li.style.borderRadius = '8px';
    li.style.background = 'var(--panel)';
    li.style.color = 'var(--muted)';
    li.innerText = '该菜谱暂无详细步骤，请自行操作或询问助手。';
    stepsList.appendChild(li);
  } else {
    currentSteps.forEach((s, idx) => {
      const li = document.createElement('li');
      li.style.padding = '10px';
      li.style.borderRadius = '8px';
      li.style.background = idx === currentStepIndex ? 'linear-gradient(90deg, rgba(47,141,230,0.06), rgba(47,141,230,0.03))' : 'var(--panel)';
      li.style.boxShadow = '0 6px 18px rgba(16,24,40,0.03)';
      li.style.marginBottom = '4px';
      li.style.listStyle = 'decimal';
      li.dataset.idx = idx;
      // 去除序号前缀，避免与 <ol> 自带的序号重复
      const cleanStep = s.replace(/^\s*[\d一二三四五六七八九十]+[.\)、\s-]*/, '').trim();
      li.innerText = cleanStep;
      stepsList.appendChild(li);
    });
  }
  stepsWrapper.appendChild(stepsList);
  container.appendChild(stepsWrapper);

  const controls = document.createElement('div');
  controls.id = 'cooking_controls';
  container.appendChild(controls);

  cookingContent.appendChild(container);
  renderCookingControls();
}

// Render cooking controls for current step
function renderCookingControls() {
  const ctrl = document.getElementById('cooking_controls');
  if (!ctrl) return;
  ctrl.innerHTML = '';

  if (!currentSteps || currentSteps.length === 0) {
    ctrl.innerHTML = '<div style="color:var(--muted)">当前菜谱未提供步骤。</div>';
    return;
  }

  const stepText = currentSteps[currentStepIndex] || '';

  const stepBox = document.createElement('div');
  stepBox.style.padding = '12px';
  stepBox.style.borderRadius = '10px';
  stepBox.style.background = 'var(--panel)';
  stepBox.style.boxShadow = '0 8px 20px rgba(16,24,40,0.04)';
  // 去除当前步骤文本的序号前缀
  const cleanStepText = stepText.replace(/^\s*[\d一二三四五六七八九十]+[.\)、\s-]*/, '').trim();
  stepBox.innerHTML = `<div style="display:flex;align-items:center;justify-content:space-between"><div style="font-weight:700">步骤 ${currentStepIndex+1}</div><div style="color:var(--muted);font-size:13px">${currentSteps.length} 步</div></div><div id="currentStepText" style="margin-top:10px;line-height:1.6">${escapeHtml(cleanStepText)}</div>`;
  ctrl.appendChild(stepBox);

  const row = document.createElement('div');
  row.style.display = 'flex';
  row.style.gap = '8px';
  row.style.marginTop = '10px';
  row.style.flexWrap = 'wrap';

  // Play / TTS
  const playBtn = document.createElement('button');
  playBtn.className = 'btn';
  playBtn.innerText = '播放当前步骤';
  playBtn.onclick = async () => {
    try {
      const resp = await fetch('/message', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ text: `请朗读以下烹饪步骤：${stepText}` }) });
      if (!resp.ok) showToast('播报请求失败');
      else {
        const j = await resp.json();
        appendAssistant(j.reply || '正在播报...');
      }
    } catch (e) { console.warn(e); showToast('播报请求出错'); }
  };
  row.appendChild(playBtn);

  // Prev / Next
  const prevBtn = document.createElement('button');
  prevBtn.className = 'btn ghost';
  prevBtn.innerText = '上一步';
  prevBtn.onclick = () => {
    stopStepTimer();
    currentStepIndex = Math.max(0, currentStepIndex - 1);
    renderCookingCard();
  };
  row.appendChild(prevBtn);

  const nextBtn = document.createElement('button');
  nextBtn.className = 'btn ghost';
  nextBtn.innerText = '下一步';
  nextBtn.onclick = () => {
    stopStepTimer();
    if (currentStepIndex < currentSteps.length - 1) {
      currentStepIndex++;
      renderCookingCard();
    } else showCookingFinished();
  };
  row.appendChild(nextBtn);

  // Timer detection and controls
  const seconds = parseDurationFromText(stepText);
  if (seconds > 0) {
    const display = document.createElement('div');
    display.id = 'stepTimerDisplay';
    display.style.fontFamily = 'monospace';
    display.style.fontSize = '18px';
    display.style.minWidth = '80px';
    display.style.textAlign = 'center';
    display.style.padding = '6px 8px';
    display.style.borderRadius = '8px';
    display.style.background = 'rgba(0,0,0,0.03)';
    display.innerText = formatSeconds(stepTimer.secondsLeft || seconds);
    row.appendChild(display);

    const startBtn = document.createElement('button');
    startBtn.className = 'btn';
    startBtn.innerText = '开始计时';
    startBtn.onclick = () => {
      if (stepTimer.running) return;
      if (!stepTimer.secondsLeft) stepTimer.secondsLeft = seconds;
      startStepTimer(() => {
        const el = document.getElementById('stepTimerDisplay'); if (el) el.innerText = formatSeconds(stepTimer.secondsLeft);
      }, () => {
        const el = document.getElementById('stepTimerDisplay'); if (el) el.innerText = '完成';
      });
    };
    row.appendChild(startBtn);

    const pauseBtn = document.createElement('button');
    pauseBtn.className = 'btn ghost';
    pauseBtn.innerText = '暂停';
    pauseBtn.onclick = pauseStepTimer;
    row.appendChild(pauseBtn);

    const resetBtn = document.createElement('button');
    resetBtn.className = 'btn ghost';
    resetBtn.innerText = '重置';
    resetBtn.onclick = () => {
      stopStepTimer();
      stepTimer.secondsLeft = seconds;
      const el = document.getElementById('stepTimerDisplay'); if (el) el.innerText = formatSeconds(stepTimer.secondsLeft);
    };
    row.appendChild(resetBtn);

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'btn';
    confirmBtn.innerText = '完成当前步骤';
    confirmBtn.onclick = () => {
      stopStepTimer();
      if (currentStepIndex < currentSteps.length - 1) {
        currentStepIndex++;
        renderCookingCard();
      } else showCookingFinished();
    };
    row.appendChild(confirmBtn);
  } else {
    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'btn';
    confirmBtn.innerText = '完成当前步骤';
    confirmBtn.onclick = () => {
      if (currentStepIndex < currentSteps.length - 1) {
        currentStepIndex++;
        renderCookingCard();
      } else showCookingFinished();
    };
    row.appendChild(confirmBtn);
  }

  ctrl.appendChild(row);
}

// Parse duration like "10分钟 30秒" etc.
function parseDurationFromText(text) {
  let s = 0;
  const h = text.match(/(\d+)\s*小时/);
  const m = text.match(/(\d+)\s*分钟/);
  const sec = text.match(/(\d+)\s*秒/);
  if (h) s += parseInt(h[1],10) * 3600;
  if (m) s += parseInt(m[1],10) * 60;
  if (sec) s += parseInt(sec[1],10);
  if (!m) {
    const m2 = text.match(/(\d+)\s*分(?!钟)/);
    if (m2) s += parseInt(m2[1],10) * 60;
  }
  return s || 0;
}

function formatSeconds(s) {
  const mm = Math.floor(s / 60).toString().padStart(2,'0');
  const ss = Math.floor(s % 60).toString().padStart(2,'0');
  return `${mm}:${ss}`;
}

// simple beep notification
function _playBeep() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = 'sine';
    o.frequency.value = 880;
    o.connect(g);
    g.connect(ctx.destination);
    g.gain.value = 0.0001;
    o.start();
    g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + 0.02);
    setTimeout(() => {
      try {
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.02);
        o.stop(ctx.currentTime + 0.05);
        setTimeout(() => { try { ctx.close(); } catch(e){} }, 100);
      } catch(e) {}
    }, 700);
  } catch (e) {}
}

// Timer functions
function startStepTimer(onTick, onFinish) {
  if (stepTimer.running) return;
  if (!stepTimer.secondsLeft || stepTimer.secondsLeft <= 0) {
    const sec = parseDurationFromText(currentSteps[currentStepIndex] || '') || 0;
    stepTimer.secondsLeft = sec;
  }
  if (!stepTimer.secondsLeft || stepTimer.secondsLeft <= 0) { if (onFinish) onFinish(); return; }
  stepTimer.running = true;
  if (stepTimer.intervalId) { clearInterval(stepTimer.intervalId); stepTimer.intervalId = null; }

  stepTimer.intervalId = setInterval(() => {
    if (!stepTimer.running) return;
    stepTimer.secondsLeft -= 1;
    if (onTick) onTick();

    const displayEl = document.getElementById('stepTimerDisplay');
    if (displayEl && stepTimer.secondsLeft <= 5 && stepTimer.secondsLeft > 0) {
      displayEl.style.transition = 'transform 120ms';
      displayEl.style.transform = 'scale(1.06)';
      setTimeout(()=>{ try { displayEl.style.transform = ''; } catch(e){} }, 220);
    }

    if (stepTimer.secondsLeft <= 0) {
      stopStepTimer();
      try { _playBeep(); } catch(e) {}
      try { if (navigator.vibrate) navigator.vibrate([200,100,200]); } catch (e) {}
      appendAssistant('计时已结束。');
      showToast('计时结束', 3000);
      if (onFinish) onFinish();

      // auto-advance
      try {
        if (currentStepIndex < currentSteps.length - 1) {
          currentStepIndex++;
          setTimeout(() => { renderCookingCard(); }, 350);
        } else {
          setTimeout(showCookingFinished, 350);
        }
      } catch (e) { console.warn(e); }
    }
  }, 1000);
}

function pauseStepTimer() {
  stepTimer.running = false;
  if (stepTimer.intervalId) { clearInterval(stepTimer.intervalId); stepTimer.intervalId = null; }
}

function stopStepTimer() {
  stepTimer.running = false;
  if (stepTimer.intervalId) { clearInterval(stepTimer.intervalId); stepTimer.intervalId = null; }
  const d = document.getElementById('stepTimerDisplay');
  if (d) {
    if (!stepTimer.secondsLeft || stepTimer.secondsLeft <= 0) d.innerText = '完成';
    else d.innerText = formatSeconds(stepTimer.secondsLeft);
  }
}

function showCookingFinished() {
  stopStepTimer();
  const ctrl = document.getElementById('cooking_controls');
  if (ctrl) {
    ctrl.innerHTML = `<div style="padding:12px;background:var(--panel);border-radius:10px">
      <div style="font-weight:700">已完成全部步骤 🎉</div>
      <div style="margin-top:8px;color:var(--muted)">烹饪完成。</div>
      <div style="margin-top:12px;display:flex;gap:8px">
        <button id="doneBack" class="btn">返回菜谱</button>
      </div>
    </div>`;
    const back = document.getElementById('doneBack');
    back && back.addEventListener('click', () => switchView('recipes'));
  }
}

  // --- Preferences UI ---
  async function loadPreferencesUI() {
    if (!settingsContent) return;
    try {
      const r = await fetch('/preferences');
      if (!r.ok) return;
      const j = await r.json();
      const prefs = j.preferences || {};
      let panel = document.getElementById('prefs_panel'); if (panel) panel.remove();
      panel = document.createElement('div'); panel.id = 'prefs_panel';
      panel.innerHTML = `<div id="prefs_list"></div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
          <select id="pref_type_select"><option value="饮食偏好">饮食偏好</option><option value="口味">口味</option><option value="禁忌">禁忌</option></select>
          <input id="pref_value_input" placeholder="例如：不吃辣" />
          <button id="pref_add_btn" class="btn">添加偏好</button>
        </div>`;
      settingsContent.appendChild(panel);
      const prefsListEl = document.getElementById('prefs_list'); prefsListEl.innerHTML = '';
      Object.keys(prefs).forEach(k => (prefs[k] || []).forEach(v => {
        const el = document.createElement('div'); el.style.display='flex'; el.style.gap='8px'; el.style.marginBottom='6px';
        el.innerHTML = `<strong style="min-width:80px">${escapeHtml(k)}</strong><span style="color:var(--muted)">${escapeHtml(v)}</span>`;
        const del = document.createElement('button'); del.className='btn ghost'; del.innerText='删除';
        del.addEventListener('click', async () => {
          try {
            const resp = await fetch('/preferences', { method: 'DELETE', headers: {'Content-Type':'application/json'}, body: JSON.stringify({pref_type: k, value: v}) });
            if (resp.ok) loadPreferencesUI(); else showToast('删除失败');
          } catch (e) { console.warn(e); showToast('请求失败'); }
        });
        el.appendChild(del); prefsListEl.appendChild(el);
      }));
      document.getElementById('pref_add_btn').addEventListener('click', async () => {
        const type = document.getElementById('pref_type_select').value;
        const val = document.getElementById('pref_value_input').value.trim();
        if (!val) { showToast('请输入偏好值'); return; }
        try {
          const resp = await fetch('/preferences', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({pref_type:type, value:val}) });
          if (resp.ok) { document.getElementById('pref_value_input').value=''; loadPreferencesUI(); } else showToast('添加失败');
        } catch (e) { console.warn(e); showToast('请求失败'); }
      });
    } catch (e) { console.warn('loadPreferencesUI error', e); }
  }

  // --- Touch drag improvements & resize ---
  const MOVE_THRESHOLD = 8;
  let touchStartY = 0, scrollStart = 0, isTouching = false, activeScrollEl = null, dragging = false;
  function isInteractiveTarget(target) {
    try {
      if (!target) return false;
      return Boolean(target.closest && (target.closest('button') || target.closest('a') || target.closest('input') || target.closest('textarea') || target.closest('select') || target.closest('[role="button"]') || target.closest('.menu-item') || target.closest('.mic-btn') || target.closest('.btn')));
    } catch (e) { return false; }
  }
  function onGlobalTouchStart(e) {
    if (!e.touches || e.touches.length !== 1) return;
    const tg = e.target;
    if (isInteractiveTarget(tg) || (tg.closest && tg.closest('.sidebar'))) { isTouching = false; activeScrollEl = null; dragging = false; return; }
    isTouching = true; dragging = false; touchStartY = e.touches[0].clientY; activeScrollEl = getCurrentScrollable(); scrollStart = activeScrollEl ? activeScrollEl.scrollTop : 0;
  }
  function onGlobalTouchMove(e) {
    if (!isTouching || !e.touches || e.touches.length !== 1) return;
    const y = e.touches[0].clientY; const dy = touchStartY - y;
    if (!dragging && Math.abs(dy) > MOVE_THRESHOLD) dragging = true;
    if (dragging) { if (e.cancelable) e.preventDefault(); if (activeScrollEl) activeScrollEl.scrollTop = scrollStart + dy; }
  }
  function onGlobalTouchEnd() { isTouching = false; dragging = false; activeScrollEl = null; }
  layout.addEventListener('touchstart', onGlobalTouchStart, { passive: false });
  layout.addEventListener('touchmove', onGlobalTouchMove, { passive: false });
  layout.addEventListener('touchend', onGlobalTouchEnd, { passive: true });
  layout.addEventListener('touchcancel', onGlobalTouchEnd, { passive: true });

  let mouseDown = false, mouseStartY = 0, mouseScrollStart = 0;
  layout.addEventListener('mousedown', (e) => {
    if (isInteractiveTarget(e.target) || (e.target.closest && e.target.closest('.sidebar'))) { mouseDown = false; return; }
    mouseDown = true; mouseStartY = e.clientY; const el = getCurrentScrollable(); mouseScrollStart = el ? el.scrollTop : 0;
  }, { passive: true });
  document.addEventListener('mousemove', (e) => {
    if (!mouseDown) return; const dy = mouseStartY - e.clientY; const el = getCurrentScrollable(); if (el) el.scrollTop = mouseScrollStart + dy;
  }, { passive: true });
  document.addEventListener('mouseup', () => { mouseDown = false; }, { passive: true });

  function getCurrentScrollable() {
    try {
      const curView = Array.from(document.querySelectorAll('.view')).find(v => {
        const s = window.getComputedStyle(v); return s.display !== 'none';
      });
      if (curView) {
        const candidate = curView.querySelector(".messages, .content, .scrollable-area, #assistant_view");
        if (candidate && candidate.scrollHeight > candidate.clientHeight) return candidate;
      }
      if (assistantView && assistantView.scrollHeight > assistantView.clientHeight) return assistantView;
    } catch (e) {}
    return mainArea || document.scrollingElement || document.documentElement;
  }

  function resizeMessages() {
    try {
      const header = document.querySelector('.main-header') || document.querySelector('.main > header');
      const status = document.getElementById('status');
      const headerH = header ? header.getBoundingClientRect().height : 0;
      const statusH = status ? status.getBoundingClientRect().height : 0;
      const availH = window.innerHeight - headerH - statusH - 28;
      document.querySelectorAll('.view .messages, .messages, .view .content').forEach(el => {
        try { if (el && el instanceof HTMLElement) { el.style.maxHeight = (availH > 120 ? availH : 120) + 'px'; el.style.overflow = 'auto'; el.style.webkitOverflowScrolling = 'touch'; el.style.overscrollBehavior = 'contain'; } } catch(e){}
      });
    } catch (e) {}
  }
  window.addEventListener('resize', resizeMessages, { passive: true });
  window.addEventListener('orientationchange', resizeMessages, { passive: true });
  setTimeout(resizeMessages, 80);

  // --- Init ---
  appendAssistant('你好！我会监听你的语音。点击麦克风开始/停止。');
  if (assistantView) {
    assistantView.style.touchAction = 'pan-y';
    assistantView.style.webkitOverflowScrolling = 'touch';
    assistantView.style.overscrollBehavior = 'contain';
  }
  // Bind menu buttons safely
  menuButtons.forEach(btn => btn.addEventListener('click', () => { const v = btn.dataset.view; if (v) switchView(v); }));
  // initial loads (defensive)
  loadInventory();
  loadRecipes();
  loadPreferencesUI();

  // expose debug API
  window._app = {
    switchView,
    loadInventory,
    loadRecipes,
    loadPreferencesUI,
    startCookingFromRecipe: (rc) => startCookingFromRecipe(rc)
  };

}); // DOMContentLoaded end