'use strict';
const $ = (selector) => document.querySelector(selector);
const app = $('#app');
const nav = $('#navigation');
const labels = ['안전', '주의', '위험', '치명'];
let routeVersion = 0, pollTimer, pollController;
const seconds = (value) => Number(value).toFixed(1);
const vehicleName = (value) => value === 'car' ? '자동차' : '오토바이';
function text(selector, value) { $(selector).textContent = value; }
function element(tag, className, content) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (content !== undefined) el.textContent = content;
  return el;
}
function show(template) { app.replaceChildren($('#' + template).content.cloneNode(true)); }
function otherVideo() {
  const link = element('a', 'secondary', '다른 영상 분석'); link.href = '#/'; nav.replaceChildren(link);
}
async function responseJSON(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error('서버 응답을 읽을 수 없습니다. 잠시 후 다시 시도해 주세요.'); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '요청 내용을 확인해 주세요.');
  return data;
}
function showError(message) {
  otherVideo();
  const main = element('main', 'analyzing-layout stack');
  main.append(element('h1', '', '분석을 표시할 수 없어요'), element('p', 'error', message));
  const retry = element('button', 'primary', '다시 확인'); retry.onclick = route;
  main.append(retry); app.replaceChildren(main);
}

async function uploadPage(version) {
  show('upload-template');
  nav.replaceChildren();
  for (const [label, hash] of [['영상 분석','#/'],['판정 원칙','#/how'],['한계','#/limits']]) {
    const a = element('a', '', label); a.href = hash; nav.append(a);
  }
  let selected = null, enabled = false;
  const form = $('#upload-form'), input = $('#video-file'), submit = $('#submit-button'), message = $('#upload-message');
  submit.disabled = true;
  function select(file) {
    selected = file;
    text('#selected-file', file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)}MB` : '');
    message.textContent = '';
  }
  input.addEventListener('change', () => select(input.files[0]));
  const dropzone = $('#dropzone');
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault(); dropzone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length !== 1) { message.textContent = '영상 한 편을 골라 주세요.'; return; }
    input.files = files; select(files[0]);
  });
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!enabled || submit.disabled) return;
    const file = selected || input.files[0];
    if (!file || !/\.(mp4|mov)$/i.test(file.name)) { message.textContent = 'MP4 또는 MOV 영상을 골라 주세요.'; return; }
    if (!file.size || file.size > 500 * 1024 * 1024) { message.textContent = '0MB보다 크고 500MB 이하인 영상을 골라 주세요.'; return; }
    const body = new FormData(); body.append('video', file); body.append('vehicle', form.elements.vehicle.value);
    submit.disabled = true; message.textContent = '영상을 올리고 있어요. 업로드가 끝날 때까지 이 창을 열어 두세요.';
    try {
      const job = await responseJSON(await fetch('/api/jobs', {method:'POST', body}));
      // Persist the address even if the user navigated while uploading.
      sessionStorage.setItem('roadscope-last-job', job.id);
      location.hash = '/job/' + job.id;
    } catch (error) {
      message.textContent = error.message; message.classList.add('error'); submit.disabled = false;
    }
  });
  try {
    const config = await responseJSON(await fetch('/api/config'));
    if (version !== routeVersion) return;
    enabled = config.uploads_enabled; submit.disabled = !enabled;
    if (!enabled) message.textContent = '현재 서버는 결과 조회 전용입니다. 새 분석은 실행하지 않습니다.';
  } catch (error) { if (version === routeVersion) message.textContent = error.message; }
  if (version === routeVersion && ['#/how', '#/limits'].includes(location.hash)) $('#' + location.hash.slice(2)).scrollIntoView();
}

function analyzing(job) {
  if (!$('#progress-number')) show('analyzing-template');
  nav.replaceChildren(element('span', 'small muted filename', `${job.filename} · ${vehicleName(job.vehicle)}`));
  text('#progress-number', job.progress + '%');
  $('#progress-fill').style.width = job.progress + '%';
  $('.progress-track').setAttribute('aria-valuenow', job.progress);
  text('#progress-note', job.status === 'queued' ? '앞선 분석이 끝나면 시작합니다.' : job.stage + (job.progress_estimated ? ' 중 · 진행률은 대략적인 값입니다.' : ' 중'));
  const info = job.video;
  if (info) text('#video-info', `${info.width}×${info.height} · ${seconds(info.fps)}fps · ${seconds(info.duration)}초`);
  text('#frame-progress', job.completed_frames ? `${job.completed_frames}장면 판정 완료` : '장면마다 위험도를 판정합니다.');
  document.querySelectorAll('.stages li').forEach((li, i) => {
    li.className = i < job.stage_index ? 'complete' : i === job.stage_index ? 'active' : '';
    li.querySelector('.step-icon').textContent = i < job.stage_index ? '✓' : String(i + 1);
    if (i === job.stage_index) li.setAttribute('aria-current', 'step'); else li.removeAttribute('aria-current');
  });
}

function renderCameras(cameras) {
  const area = $('#cameras');
  if (!cameras.length) { area.append(element('p', 'small muted camera-group', '이 영상에는 영상 단위 카메라 측정 기록이 없습니다.')); return; }
  const number = (v, digits, suffix) => v == null ? '기록 없음' : Number(v).toFixed(digits) + suffix;
  const sources = {foe:'광류',lane:'차선',vehicle_regression:'차량',default:'기본값',vehicle:'차량',
    'vehicle+lane':'차량·차선','vehicle(lane_disagree)':'차량 기준 · 차선과 차이 있음',
    'lane(vehicle_disagree)':'차선 기준 · 차량과 차이 있음','prior_600px@1280':'기본값'};
  for (const c of cameras) {
    const group = element('section', 'camera-group');
    if (cameras.length > 1) group.append(element('h2', 'small', `구간 ${c.segment + 1} · ${seconds(c.start)}~${seconds(c.end)}초`));
    const dl = element('dl', 'camera-grid');
    const source = (value) => value ? ' · ' + (sources[value] || value) : '';
    let baseline = c.lane_baseline_status === 'not_applied' ? '미적용' : '기록 없음';
    if (c.lane_baseline_m != null) baseline = c.lane_baseline_m === 0 ? '가운데 0.00 m' : `${c.lane_baseline_m > 0 ? '오른쪽' : '왼쪽'} ${Math.abs(c.lane_baseline_m).toFixed(2)} m`;
    if (c.ldw_measurable === false || c.lane_baseline_status === 'failed') baseline = '영점 측정 실패 · 이 구간 차선 편향 측정 불가';
    const values = [
      ['수평선 위치', number(c.horizon_ratio, 3, 'H') + source(c.horizon_source)],
      ['소실점 가로 위치', number(c.vp_x, 1, ' px') + (c.vp_offset_px == null ? '' : ` · 중앙 대비 ${c.vp_offset_px > 0 ? '+' : ''}${c.vp_offset_px.toFixed(1)} px`)],
      ['카메라 높이', number(c.camera_height_m, 2, ' m') + source(c.camera_height_source)],
      ['초점거리', number(c.focal_length_px, 0, ' px') + source(c.focal_length_source)],
      ['내 차 영역', number(c.ego_area_percent, 1, '%') + ` · 실내 박스 ${c.excluded_boxes}개 제외`],
      ['차로 안 평소 위치', baseline],
    ];
    for (const [name, value] of values) { const div = element('div'); div.append(element('dt', '', name), element('dd', '', value)); dl.append(div); }
    group.append(dl); area.append(group);
  }
}

function resultPage(job) {
  show('result-template'); otherVideo();
  const s = job.summary, files = job.files || {};
  document.title = job.filename + ' 분석 결과 · RoadScope';
  text('#result-title', job.filename + ' 분석 결과');
  text('#result-subtitle', `${vehicleName(job.vehicle)} · ${seconds(s.duration)}초${s.duration_source.startsWith('estimated') ? ' (길이 추정)' : ''} · ${s.frame_count}장면`);
  $('#confidence-banner').hidden = !s.confidence.low;
  $('#peak-card').classList.add('level-' + s.peak.grade);
  text('#peak-label', s.peak.label); text('#peak-detail', `${seconds(s.peak.time)}초 · ${s.peak.reason}`);
  text('#danger-count', s.danger.count + '곳');
  text('#danger-detail', (s.danger.count === 1 ? `${seconds(s.danger.intervals[0].start)}초~${seconds(s.danger.intervals[0].end)}초 · ` : '합계 ') + seconds(s.danger.duration) + '초');
  text('#caution-count', s.caution.count + '곳'); text('#caution-detail', `합계 ${seconds(s.caution.duration)}초`);
  text('#confidence-label', s.confidence.label); $('#confidence-label').style.color = s.confidence.low ? '#8A5A00' : '#1F4FD1';
  text('#confidence-note', s.confidence.note);
  const video = $('#hud-video'), seek = $('#time-seek'), timeline = $('#timeline'), events = $('#events');
  let pendingTime = null;
  seek.max = s.duration;
  function updateTime(t) {
    t = Math.min(s.duration, Math.max(0, t));
    let seg = s.segments.find(seg => t >= seg.start && t < seg.end);
    if (!seg && t >= s.duration) seg = s.segments.at(-1);
    const badge = $('#current-grade');
    if (!badge) return;
    badge.textContent = seg ? seg.label : '미평가'; badge.className = 'grade-badge' + (seg ? ' level-' + seg.grade : '');
    text('#current-time', `${seconds(t)}s / ${seconds(s.duration)}s`);
    $('#playhead').style.left = (t / s.duration * 100) + '%'; seek.value = t;
    let activeTime = -1;
    for (const event of s.events) if (event.time <= t + 0.001) activeTime = event.time;
    for (const row of events.children) row.classList.toggle('active', Number(row.dataset.time) === activeTime);
  }
  function go(t) {
    if (files.hud) {
      if (video.readyState >= 1) video.currentTime = Math.min(t, Number.isFinite(video.duration) ? video.duration : t);
      else pendingTime = t;
    }
    updateTime(t);
  }
  if (files.hud) {
    video.src = files.hud; video.hidden = false; $('#video-empty').hidden = true;
    video.addEventListener('loadedmetadata', () => { if (pendingTime !== null) { video.currentTime = Math.min(pendingTime, video.duration); pendingTime = null; } });
    video.addEventListener('timeupdate', () => updateTime(video.currentTime));
    video.addEventListener('seeked', () => updateTime(video.currentTime));
    video.addEventListener('error', () => { video.hidden = true; $('#video-empty').hidden = false; text('#video-empty', '결과 영상을 재생하지 못했습니다. 파일을 내려받아 확인해 주세요.'); });
  }
  for (const seg of s.segments) {
    const button = element('button', 'level-' + seg.grade);
    button.type = 'button'; button.style.left = (seg.start / s.duration * 100) + '%'; button.style.width = ((seg.end - seg.start) / s.duration * 100) + '%';
    button.setAttribute('aria-label', `${seg.label} ${seconds(seg.start)}초부터 ${seconds(seg.end)}초까지`); button.title = button.getAttribute('aria-label');
    button.onclick = () => go(seg.start); timeline.append(button);
  }
  for (const event of s.events) {
    const row = element('button', 'event-row'); row.type = 'button'; row.dataset.time = event.time;
    row.append(element('span', 'event-time', seconds(event.time) + 's'), element('span', 'event-pill level-' + event.grade, event.label), element('span', 'event-reason', event.reason));
    row.onclick = () => go(event.time); events.append(row);
  }
  if (!s.events.length) events.append(element('p', 'muted small', '이 영상에는 기록된 경고가 없습니다.'));
  seek.addEventListener('input', () => go(Number(seek.value)));
  renderCameras(s.cameras);
  for (const [key, label] of [['hud','결과 영상 (MP4)'],['frames','프레임별 판정 기록 (JSONL)'],['timeline','타임라인 이미지 (PNG)']]) {
    const el = element(files[key] ? 'a' : 'span', files[key] ? '' : 'unavailable', label + (files[key] ? '' : ' · 파일 없음'));
    if (files[key]) { el.href = files[key] + '?attachment=true'; el.setAttribute('download', ''); }
    $('#downloads').append(el);
  }
  updateTime(0);
}

async function poll(id, version) {
  pollController = new AbortController();
  try {
    const response = await fetch('/api/jobs/' + encodeURIComponent(id), {signal:pollController.signal, cache:'no-store'});
    if (response.status === 404) { if (version === routeVersion) showError('해당 작업을 찾을 수 없습니다. 주소를 확인해 주세요.'); return; }
    const job = await responseJSON(response);
    if (version !== routeVersion) return;
    if (job.status === 'done') { resultPage(job); return; }
    if (['error','interrupted'].includes(job.status)) { showError(job.error || '분석이 중단됐습니다.'); return; }
    analyzing(job); text('#connection-message', '');
  } catch (error) {
    if (version !== routeVersion || error.name === 'AbortError') return;
    if ($('#connection-message')) text('#connection-message', '서버 연결을 확인하고 있어요. 잠시 후 자동으로 다시 시도합니다.');
  }
  if (version === routeVersion) pollTimer = setTimeout(() => poll(id, version), 2000);
}
function route() {
  const version = ++routeVersion; clearTimeout(pollTimer); pollController?.abort();
  const oldVideo = $('#hud-video'); if (oldVideo) { oldVideo.pause(); oldVideo.removeAttribute('src'); oldVideo.load(); }
  const hash = location.hash || '#/';
  const match = hash.match(/^#\/job\/([0-9a-f]{32})$/);
  document.title = 'RoadScope · 블랙박스 영상 위험도 평가';
  if (match) { show('analyzing-template'); otherVideo(); poll(match[1], version); }
  else if (['#/','#/how','#/limits'].includes(hash)) uploadPage(version);
  else showError('올바르지 않은 주소입니다. 영상 분석 화면에서 다시 시작해 주세요.');
  window.scrollTo(0, 0);
}
window.addEventListener('hashchange', route);
route();
