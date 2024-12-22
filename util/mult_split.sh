#!/bin/bash

# 最大并发窗口数量
MAX_WINDOWS=10

# tmux 会话名称
SESSION_NAME="split_n"

# 创建一个新的 tmux 会话
tmux new-session -d -s $SESSION_NAME

# 循环从 16 到 1000
for n in $(seq 16 1000)
do
    CMD="./split 1577808000000046 1580486400000000 $n"

    # 等待空闲窗口
    while true; do
        # 获取当前窗口数量，处理空输出
        PANE_COUNT=$(tmux list-panes -t $SESSION_NAME 2>/dev/null | wc -l)
        PANE_COUNT=${PANE_COUNT:-0}  # 如果为空，则赋值为 0
        
        # 调试信息
        echo "当前窗口数量: $PANE_COUNT"
        
        if [ "$PANE_COUNT" -lt "$MAX_WINDOWS" ]; then
            break  # 如果窗口数量小于最大限制，跳出循环
        fi
        sleep 1  # 每秒检查一次
    done

    # 在会话中创建新窗口并执行命令
    tmux split-window -t $SESSION_NAME
    tmux send-keys -t $SESSION_NAME "$CMD" C-m
    tmux select-layout -t $SESSION_NAME tiled  # 保持窗口布局整齐
done

# 等待所有命令执行完毕
while [ $(tmux list-panes -t $SESSION_NAME | wc -l) -gt 0 ]; do
    sleep 1
done

# 附加到 tmux 会话
tmux attach-session -t $SESSION_NAME
