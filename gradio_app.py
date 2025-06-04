"""
圖形化使用者介面 - 使用Gradio建立AI助理的網頁介面
"""

import os
import time
import uuid
import tempfile
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import gradio as gr

# 導入所有模組
import config
from utils import setup_logger
from whisper_module import WhisperModule
from context_module import ContextModule
from rag_module import RAGModule
from mcp_module import MCPModule
from emotion_module import EmotionModule
from response_module import ResponseModule
from voice_module import VoiceModule
from memory import MemoryManager

# 設置日誌
logger = setup_logger("gradio_interface", config.LOGGING_CONFIG["level"])

class GradioInterface:
    """
    使用Gradio建立AI助理的圖形化使用者介面
    整合所有模組功能，提供語音輸入輸出、文件上傳、對話記憶等功能
    """
    
    def __init__(self):
        """
        初始化Gradio介面和所有模組
        """
        logger.info("初始化Gradio介面")
        
        # 創建臨時目錄
        self.temp_dir = Path(tempfile.mkdtemp())
        self.audio_dir = self.temp_dir / "audio"
        self.upload_dir = self.temp_dir / "uploads"
        self.audio_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        
        # 初始化模組
        self._init_modules()
        
        # 初始化對話記憶
        self.memory_manager = MemoryManager()
        self.conversation_history = []
        # Gradio Chatbot 顯示格式: [[user, assistant], ...]
        self.chat_history_pairs = []
        
        # 初始化狀態
        self.current_speaker = None
        self.processing_lock = threading.Lock()
        
        # 創建Gradio介面
        self._create_interface()
    
    def _init_modules(self):
        """
        初始化所有AI模組
        """
        try:
            logger.info("初始化AI模組")
            
            # 語音輸入模組
            self.whisper_module = WhisperModule()
            
            # 上下文解答模組
            self.context_module = ContextModule()
            
            # RAG檢索生成模組
            self.rag_module = RAGModule()
            
            # MCP工具使用模組
            self.mcp_module = MCPModule()
            
            # 情緒辨識模組
            self.emotion_module = EmotionModule()
            
            # 回應整合模組
            self.response_module = ResponseModule()
            
            # 語音生成模組
            self.voice_module = VoiceModule()
            
            logger.info("所有AI模組初始化完成")
        except Exception as e:
            logger.error(f"初始化AI模組時發生錯誤: {str(e)}")
            raise
    
    def _create_interface(self):
        """
        創建Gradio介面
        """
        try:
            # 定義介面主題和標題
            theme = gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="blue",
            )
            
            # 創建Gradio區塊
            self.interface = gr.Blocks(
                title=config.UI_CONFIG["gradio"]["title"],
                theme=theme,
                css=self._get_custom_css()
            )
            
            with self.interface:
                gr.Markdown(f"# {config.UI_CONFIG['gradio']['title']}")
                gr.Markdown(config.UI_CONFIG["gradio"]["description"])
                
                with gr.Tabs():
                    # 對話頁籤
                    with gr.TabItem("對話"):
                        self._create_chat_tab()
                    
                    # 文件管理頁籤
                    with gr.TabItem("文件管理"):
                        self._create_document_tab()
                    
                    # 語音設定頁籤
                    with gr.TabItem("語音設定"):
                        self._create_voice_tab()
                    
                    # 系統設定頁籤
                    with gr.TabItem("系統設定"):
                        self._create_settings_tab()
            
            logger.info("Gradio介面創建完成")
        except Exception as e:
            logger.error(f"創建Gradio介面時發生錯誤: {str(e)}")
            raise
    
    def _get_custom_css(self) -> str:
        """
        獲取自定義CSS樣式
        
        Returns:
            CSS樣式字符串
        """
        return """
        .message-bubble {
            padding: 10px 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .user-message {
            background-color: #e1f5fe;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #f5f5f5;
            margin-right: auto;
        }
        .emotion-tag {
            font-size: 0.8em;
            color: #666;
            font-style: italic;
            margin-left: 5px;
        }
        .thinking-process {
            font-size: 0.9em;
            color: #555;
            background-color: #f9f9f9;
            padding: 8px;
            border-left: 3px solid #ddd;
            margin-top: 5px;
        }
        """
    
    def _create_chat_tab(self):
        """
        創建對話頁籤
        """
        with gr.Row():
            with gr.Column(scale=3):
                # 對話歷史
                self.chatbot = gr.Chatbot(
                    label="對話歷史",
                    height=500,
                    show_label=True,
                    elem_id="chatbot",
                    type='messages'
                )
                # 思考過程顯示
                self.thinking_box = gr.Textbox(
                    label="思考過程",
                    lines=3,
                    interactive=False,
                    visible=False
                )
                
                # 輸入區域
                with gr.Row():
                    self.text_input = gr.Textbox(
                        label="文字輸入",
                        placeholder="在此輸入訊息...",
                        lines=2,
                        scale=8
                    )
                    
                    self.audio_input = gr.Audio(
                        label="語音輸入",
                        type="numpy",
                        scale=2
                    )
                
                # 控制按鈕
                with gr.Row():
                    self.submit_btn = gr.Button("發送", variant="primary")
                    self.clear_btn = gr.Button("清除對話")
                    self.show_thinking_btn = gr.Checkbox(label="顯示思考過程", value=False)
                
                # 回應音頻播放器
                self.audio_output = gr.Audio(
                    label="語音回應",
                    type="filepath",
                    interactive=False,
                    autoplay=True
                )
            
            with gr.Column(scale=1):
                # 狀態顯示
                self.status = gr.Textbox(
                    label="狀態",
                    value="就緒",
                    interactive=False
                )
                
                # 情緒分析結果
                self.emotion_chart = gr.Label(label="情緒分析")
                
                # 處理時間
                self.process_time = gr.Textbox(
                    label="處理時間",
                    value="0.0 秒",
                    interactive=False
                )
        
        # 事件綁定
        self.submit_btn.click(
            fn=self.process_input,
            inputs=[self.text_input],
            outputs=[self.chatbot, self.text_input, self.audio_output, self.emotion_chart, self.process_time, self.thinking_box, self.status]
        )
        
        self.audio_input.stop_recording(
            fn=self.process_audio_input,
            inputs=[self.audio_input],
            outputs=[self.chatbot, self.text_input, self.audio_output, self.emotion_chart, self.process_time, self.thinking_box, self.status]
        )
        
        self.clear_btn.click(
            fn=self.clear_chat,
            inputs=[],
            outputs=[self.chatbot, self.thinking_box, self.emotion_chart, self.process_time, self.status]
        )
        
        self.show_thinking_btn.change(
            fn=self.toggle_thinking_display,
            inputs=[self.show_thinking_btn],
            outputs=[self.thinking_box]
        )
    
    def _create_document_tab(self):
        """
        創建文件管理頁籤
        """
        with gr.Row():
            with gr.Column():
                # 文件上傳
                self.file_upload = gr.File(
                    label="上傳文件 (支援 TXT, PDF)",
                    file_types=[".txt", ".pdf"],
                    file_count="multiple"
                )
                
                # 上傳按鈕
                self.upload_btn = gr.Button("上傳並處理文件", variant="primary")
                
                # 文件統計
                self.file_stats = gr.JSON(label="文件統計")
                
                # 清除文件按鈕
                with gr.Row():
                    self.clear_file_btn = gr.Button("清除所有文件")
                    self.refresh_stats_btn = gr.Button("刷新統計")
        
        # 事件綁定
        self.upload_btn.click(
            fn=self.process_uploaded_files,
            inputs=[self.file_upload],
            outputs=[self.file_stats, self.status]
        )
        
        self.clear_file_btn.click(
            fn=self.clear_all_files,
            inputs=[],
            outputs=[self.file_stats, self.status]
        )
        
        self.refresh_stats_btn.click(
            fn=self.refresh_file_stats,
            inputs=[],
            outputs=[self.file_stats]
        )
    
    def _create_voice_tab(self):
        """
        創建語音設定頁籤
        """
        with gr.Row():
            with gr.Column():
                # 語音克隆
                gr.Markdown("### 語音克隆")
                
                # 上傳參考音頻
                self.voice_upload = gr.Audio(
                    label="上傳參考音頻",
                    type="filepath"
                )
                
                # 說話者名稱
                self.speaker_name = gr.Textbox(
                    label="說話者名稱",
                    placeholder="輸入一個名稱以保存語音特徵"
                )
                
                # 克隆按鈕
                self.clone_btn = gr.Button("克隆語音", variant="primary")
                
                # 已保存的說話者
                self.saved_speakers = gr.Dropdown(
                    label="選擇說話者",
                    choices=self.get_saved_speakers(),
                    value=None
                )
                
                # 刷新說話者列表按鈕
                self.refresh_speakers_btn = gr.Button("刷新說話者列表")
        
        # 事件綁定
        self.clone_btn.click(
            fn=self.clone_voice,
            inputs=[self.voice_upload, self.speaker_name],
            outputs=[self.saved_speakers, self.status]
        )
        
        self.refresh_speakers_btn.click(
            fn=self.refresh_speakers,
            inputs=[],
            outputs=[self.saved_speakers]
        )
        
        self.saved_speakers.change(
            fn=self.select_speaker,
            inputs=[self.saved_speakers],
            outputs=[self.status]
        )
    
    def _create_settings_tab(self):
        """
        創建系統設定頁籤
        """
        with gr.Row():
            with gr.Column():
                # 系統設定
                gr.Markdown("### 系統設定")
                
                # 顯示思考過程
                self.always_show_thinking = gr.Checkbox(
                    label="總是顯示思考過程",
                    value=False
                )
                
                # 自動播放語音
                self.autoplay_audio = gr.Checkbox(
                    label="自動播放語音回應",
                    value=True
                )
                
                # 保存設定按鈕
                self.save_settings_btn = gr.Button("保存設定", variant="primary")
        
        # 事件綁定
        self.save_settings_btn.click(
            fn=self.save_settings,
            inputs=[self.always_show_thinking, self.autoplay_audio],
            outputs=[self.status]
        )
    
    def process_input(self, text_input: str) -> Tuple:
        """
        處理文字輸入
        
        Args:
            text_input: 用戶輸入文字
            
        Returns:
            更新後的UI元素元組
        """
        # 檢查輸入是否為空
        if not text_input or text_input.strip() == "":
            return (
                self.chatbot,
                "",
                None,
                None,
                "0.0 秒",
                "",
                "輸入不能為空"
            )
        
        # 獲取處理鎖
        if not self.processing_lock.acquire(blocking=False):
            return (
                self.chatbot,
                text_input,
                None,
                None,
                "0.0 秒",
                "",
                "正在處理上一個請求，請稍候"
            )
        
        try:
            # 更新狀態
            status = "正在處理..."
            
            # 記錄開始時間
            start_time = time.time()
            
            # 添加用戶消息到對話歷史
            user_message = {"role": "user", "content": text_input, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.conversation_history.append(user_message)
            # 更新聊天界面 (Gradio格式)
            self.chat_history_pairs.append([text_input, None])
            
            # 分析輸入情緒
            emotion_distribution = self.emotion_module.analyze_input_emotion(text_input)
            
            # 處理上下文解答
            context_answer, reasoning = self.context_module.generate_answer(
                text_input, 
                self.conversation_history
            )
            
            # 處理RAG檢索生成
            rag_summary = self.rag_module.process_query(text_input)
            
            # 處理MCP工具使用
            mcp_result = self.mcp_module.process_query(text_input)
            
            # 生成最終回應
            response, thought_process = self.response_module.chain_of_thought_response(
                text_input,
                context_answer,
                rag_summary,
                mcp_result,
                emotion_distribution,
                self.conversation_history
            )
            
            # 添加情緒標籤
            tagged_response = self.emotion_module.add_emotion_tags(response)
            
            # 生成語音回應
            audio_output_path = self.audio_dir / f"response_{uuid.uuid4()}.wav"
            
            speaker_embedding = None
            if self.current_speaker:
                speaker_embedding = self.voice_module.load_speaker_embedding(self.current_speaker)
            
            self.voice_module.emotional_tts(
                tagged_response,
                audio_output_path,
                speaker_embedding
            )
            
            # 添加助理消息到對話歷史
            assistant_message = {"role": "assistant", "content": response, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.conversation_history.append(assistant_message)
            # 更新聊天界面 (Gradio格式)
            if self.chat_history_pairs:
                self.chat_history_pairs[-1][1] = response
            
            # 計算處理時間
            end_time = time.time()
            process_time = f"{end_time - start_time:.2f} 秒"
            
            # 格式化思考過程
            thinking_text = ""
            for key, value in thought_process.items():
                thinking_text += f"**{key}**: {value}\n\n"
            
            # 更新狀態
            status = "處理完成"
            
            return (
                self.chat_history_pairs.copy(),
                "",
                str(audio_output_path),
                {"label": emotion_distribution},
                process_time,
                thinking_text,
                status
            )
        except Exception as e:
            logger.error(f"處理輸入時發生錯誤: {str(e)}")
            return (
                self.chat_history_pairs.copy(),
                text_input,
                None,
                None,
                "0.0 秒",
                "",
                f"處理錯誤: {str(e)}"
            )
        finally:
            # 釋放處理鎖
            self.processing_lock.release()
    
    def process_audio_input(self, audio_input) -> Tuple:
        """
        處理語音輸入
        
        Args:
            audio_input: 用戶語音輸入
            
        Returns:
            更新後的UI元素元組
        """
        try:
            # 檢查輸入是否為空
            if audio_input is None:
                return (
                    self.chatbot,
                    "",
                    None,
                    None,
                    "0.0 秒",
                    "",
                    "未檢測到語音輸入"
                )
            
            # 更新狀態
            status = "正在處理語音輸入..."
            
            # 將語音轉為文字
            audio_array = audio_input # (樣本率, 音頻數據)
            sample_rate = 16000
            
            transcription = self.whisper_module.transcribe_audio(
                audio_array=audio_array,
                sample_rate=sample_rate
            )
            
            # 更新文字輸入框
            text_input = transcription
            
            # 使用文字處理函數處理轉錄文字
            return self.process_input(text_input)
        except Exception as e:
            logger.error(f"處理語音輸入時發生錯誤: {str(e)}")
            return (
                self.chatbot,
                "",
                None,
                None,
                "0.0 秒",
                "",
                f"處理語音錯誤: {str(e)}"
            )
    
    def clear_chat(self) -> Tuple:
        """
        清除對話歷史
        
        Returns:
            更新後的UI元素元組
        """
        # 清除對話歷史
        self.conversation_history = []
        self.chat_history_pairs = []
        return (
            [],
            "",
            None,
            "0.0 秒",
            "對話已清除"
        )
    
    def toggle_thinking_display(self, show_thinking: bool) -> gr.Textbox:
        """
        切換思考過程顯示
        
        Args:
            show_thinking: 是否顯示思考過程
            
        Returns:
            更新後的思考過程文本框
        """
        return gr.Textbox(visible=show_thinking)
    
    def process_uploaded_files(self, files) -> Tuple:
        """
        處理上傳的文件
        
        Args:
            files: 上傳的文件列表
            
        Returns:
            更新後的UI元素元組
        """
        if not files:
            return self.file_stats, "未選擇文件"
        
        try:
            # 更新狀態
            status = "正在處理文件..."
            
            # 處理每個文件
            for file in files:
                file_path = Path(file.name)
                
                # 將文件添加到RAG
                self.rag_module.add_file(file_path)
            
            # 獲取文件統計
            stats = self.rag_module.get_file_statistics()
            
            return stats, f"成功處理 {len(files)} 個文件"
        except Exception as e:
            logger.error(f"處理上傳文件時發生錯誤: {str(e)}")
            return self.file_stats, f"處理文件錯誤: {str(e)}"
    
    def clear_all_files(self) -> Tuple:
        """
        清除所有文件
        
        Returns:
            更新後的UI元素元組
        """
        try:
            # 清除RAG中的文件數據
            removed_count = self.rag_module.clear_file_data()
            
            # 獲取文件統計
            stats = self.rag_module.get_file_statistics()
            
            return stats, f"成功清除 {removed_count} 個文件塊"
        except Exception as e:
            logger.error(f"清除文件時發生錯誤: {str(e)}")
            return self.file_stats, f"清除文件錯誤: {str(e)}"
    
    def refresh_file_stats(self) -> Dict:
        """
        刷新文件統計
        
        Returns:
            文件統計字典
        """
        try:
            return self.rag_module.get_file_statistics()
        except Exception as e:
            logger.error(f"刷新文件統計時發生錯誤: {str(e)}")
            return {"error": str(e)}
    
    def get_saved_speakers(self) -> List[str]:
        """
        獲取已保存的說話者列表
        
        Returns:
            說話者名稱列表
        """
        try:
            speaker_dir = self.voice_module.speaker_embedding_path
            if not speaker_dir.exists():
                return []
            
            speakers = []
            for file in speaker_dir.glob("*.npy"):
                speakers.append(file.stem)
            
            return speakers
        except Exception as e:
            logger.error(f"獲取說話者列表時發生錯誤: {str(e)}")
            return []
    
    def clone_voice(self, voice_file: str, speaker_name: str) -> Tuple:
        """
        克隆語音
        
        Args:
            voice_file: 語音文件路徑
            speaker_name: 說話者名稱
            
        Returns:
            更新後的UI元素元組
        """
        if not voice_file or not speaker_name:
            return self.get_saved_speakers(), "語音文件或說話者名稱不能為空"
        
        try:
            # 提取說話者嵌入
            embedding = self.voice_module.extract_speaker_embedding(voice_file)
            
            if embedding is None:
                return self.get_saved_speakers(), "提取語音特徵失敗"
            
            # 保存說話者嵌入
            success = self.voice_module.save_speaker_embedding(speaker_name, embedding)
            
            if not success:
                return self.get_saved_speakers(), "保存語音特徵失敗"
            
            # 設置當前說話者
            self.current_speaker = speaker_name
            
            return self.get_saved_speakers(), f"成功克隆語音: {speaker_name}"
        except Exception as e:
            logger.error(f"克隆語音時發生錯誤: {str(e)}")
            return self.get_saved_speakers(), f"克隆語音錯誤: {str(e)}"
    
    def refresh_speakers(self) -> List[str]:
        """
        刷新說話者列表
        
        Returns:
            說話者名稱列表
        """
        return self.get_saved_speakers()
    
    def select_speaker(self, speaker_name: str) -> str:
        """
        選擇說話者
        
        Args:
            speaker_name: 說話者名稱
            
        Returns:
            狀態消息
        """
        if not speaker_name:
            self.current_speaker = None
            return "已清除說話者選擇"
        
        self.current_speaker = speaker_name
        return f"已選擇說話者: {speaker_name}"
    
    def save_settings(self, always_show_thinking: bool, autoplay_audio: bool) -> str:
        """
        保存設定
        
        Args:
            always_show_thinking: 是否總是顯示思考過程
            autoplay_audio: 是否自動播放語音
            
        Returns:
            狀態消息
        """
        try:
            # 更新設定
            # 這裡可以添加將設定保存到文件的代碼
            
            # 更新UI
            self.thinking_box.visible = always_show_thinking
            self.audio_output.autoplay = autoplay_audio
            
            return "設定已保存"
        except Exception as e:
            logger.error(f"保存設定時發生錯誤: {str(e)}")
            return f"保存設定錯誤: {str(e)}"
    
    def launch(self, share: bool = False):
        """
        啟動Gradio介面
        
        Args:
            share: 是否生成公共鏈接
        """
        try:
            port = config.UI_CONFIG["gradio"]["port"]
            self.interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                inbrowser=True
            )
        except Exception as e:
            logger.error(f"啟動Gradio介面時發生錯誤: {str(e)}")
            raise

# 主函數
def main():
    # 創建並啟動Gradio介面
    interface = GradioInterface()
    interface.launch()

if __name__ == "__main__":
    main()
