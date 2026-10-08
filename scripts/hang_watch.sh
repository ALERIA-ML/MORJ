#!/usr/bin/env bash
# Watch a running service and record evidence when it stops answering /health.
#
#   scripts/hang_watch.sh [CONTAINER] [URL] [OUT_DIR]
#   defaults: morj-server-1  http://127.0.0.1:30104  runs/hangs
#
# Waits until /health answers once (a loading service is never signalled: before the engine is up,
# SIGUSR1 would terminate it), then polls /health every 5 s. After 3 failures in a row (about 15 s
# without an answer) it writes one
# folder per episode with: every Python thread's stack (SIGUSR1 -> faulthandler, written without the GIL
# or stdout), each thread's kernel wait channel and state, process status, nvidia-smi, container state
# and resource use, and recent kernel / journal lines when readable. When /health answers again it
# records how long the service was unresponsive. Read-only: it never restarts or changes the service.
set -u
C=${1:-morj-server-1}; URL=${2:-http://127.0.0.1:30104}; OUT=${3:-runs/hangs}
mkdir -p "$OUT"; LOG=$OUT/watch.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
arm() {
  log "waiting for $URL/health before arming (the service may still be loading)"
  until curl -fsS -m 5 "$URL/health" > /dev/null 2>&1; do sleep 10; done
  armed_start=$(docker inspect -f '{{.State.StartedAt}}' "$C" 2>/dev/null)
  log "armed: watching $C at $URL (container started $armed_start)"
}
arm
fails=0; episode=''
while true; do
  if curl -fsS -m 5 "$URL/health" > /dev/null 2>&1; then
    if [ -n "$episode" ]; then
      log "recovered after $(( $(date +%s) - start )) s (episode $episode)"; episode=''
    fi
    fails=0
  else
    fails=$((fails + 1))
    if [ "$fails" -eq 3 ] && [ -z "$episode" ] && [ "$(docker inspect -f '{{.State.StartedAt}}' "$C" 2>/dev/null)" != "$armed_start" ]; then
      log "container restarted or gone since arming: not signalling a loading service"; arm; fails=0; continue
    fi
    if [ "$fails" -eq 3 ] && [ -z "$episode" ]; then
      start=$(( $(date +%s) - 15 )); episode=$(date -u +%Y%m%dT%H%M%SZ); D=$OUT/$episode; mkdir -p "$D"
      log "no /health answer for ~15 s: capturing $D"
      docker inspect -f '{{.State.Status}} pid={{.State.Pid}} restarts={{.RestartCount}} health={{.State.Health.Status}}' "$C" > "$D/container.txt" 2>&1
      docker kill -s USR1 "$C" > /dev/null 2>&1; sleep 3
      timeout 20 docker exec "$C" sh -c 'tail -n 2000 /tmp/morj-stacks.log' > "$D/python-stacks.txt" 2>&1
      timeout 20 docker exec "$C" sh -c '
        for p in /proc/[0-9]*; do
          grep -q morj "$p/cmdline" 2>/dev/null || continue
          echo "== pid ${p#/proc/}: $(tr "\0" " " < $p/cmdline)"; grep -E "^(State|Threads|VmRSS)" $p/status
          for t in $p/task/*; do
            echo "tid ${t##*/} $(cat $t/comm) state=$(cut -d" " -f3 $t/stat) wchan=$(cat $t/wchan 2>/dev/null)"
          done
        done' > "$D/threads.txt" 2>&1
      timeout 20 nvidia-smi -q > "$D/nvidia-smi.txt" 2>&1
      timeout 20 docker stats --no-stream "$C" > "$D/docker-stats.txt" 2>&1
      timeout 20 docker logs --tail 50 "$C" > "$D/service-tail.txt" 2>&1
      (timeout 20 journalctl -k --since '-15 min' --no-pager 2>/dev/null; timeout 20 dmesg -T 2>/dev/null | tail -n 100) > "$D/kernel.txt"
      timeout 20 journalctl --since '-15 min' --no-pager > "$D/journal.txt" 2>&1
      log "captured $D: $(wc -l < "$D/python-stacks.txt") stack lines, $(grep -c '^tid' "$D/threads.txt") threads"
    fi
  fi
  sleep 5
done
