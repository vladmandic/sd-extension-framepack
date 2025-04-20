
function submit_framepack(...args) {
  const id = randomId();
  log('submitFramepack', id);
  requestProgress(id, null, null);
  window.submit_state = '';
  args[0] = id;
  return args;
}
