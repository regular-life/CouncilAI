FROM vllm/vllm-openai:latest

ENV HF_HOME=/data/huggingface

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8001
ENTRYPOINT ["/start.sh"]
