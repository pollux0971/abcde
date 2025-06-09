"""
命令列介面 - 提供AI助理的終端互動方式
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
logger = setup_logger("cli_interface", config.LOGGING_CONFIG["level"])

class CLIInterface:
    """
    提供AI助理的命令列介面，支援文字輸入輸出、語音互動和文件處理
    """
    
    def __init__(self, enable_mcp: bool = False):
        """
        初始化命令列介面和所有模組
        """
        logger.info("初始化命令列介面")
        self.enable_mcp = enable_mcp
        
        # 設置顏色代碼 (必須先設置，因為 _init_modules 會用到)
        self.colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "user": "\033[94m",  # 藍色
            "assistant": "\033[92m",  # 綠色
            "system": "\033[93m",  # 黃色
            "error": "\033[91m",  # 紅色
            "thinking": "\033[90m"  # 灰色
        }

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

        # 初始化狀態
        self.current_speaker = None
        self.show_thinking = False
        self.autoplay_audio = True
    
    def _init_modules(self):
        """
        初始化所有AI模組
        """
        try:
            logger.info("初始化AI模組")
            print(f"{self.colors['system']}正在初始化AI模組，請稍候...{self.colors['reset']}")
            
            # 語音輸入模組
            self.whisper_module = WhisperModule()
            print(f"{self.colors['system']}✓ 語音輸入模組已載入{self.colors['reset']}")
            
            # 上下文解答模組
            self.context_module = ContextModule()
            print(f"{self.colors['system']}✓ 上下文解答模組已載入{self.colors['reset']}")
            
            # RAG檢索生成模組
            self.rag_module = RAGModule()
            print(f"{self.colors['system']}✓ RAG檢索生成模組已載入{self.colors['reset']}")
            
            if self.enable_mcp:
                # MCP工具使用模組
                self.mcp_module = MCPModule()
                print(f"{self.colors['system']}✓ MCP工具使用模組已載入{self.colors['reset']}")
            else:
                print(f"{self.colors['system']}✓ MCP工具使用模組已禁用{self.colors['reset']}")
            
            # 情緒辨識模組
            self.emotion_module = EmotionModule()
            print(f"{self.colors['system']}✓ 情緒辨識模組已載入{self.colors['reset']}")
            
            # 回應整合模組
            self.response_module = ResponseModule()
            print(f"{self.colors['system']}✓ 回應整合模組已載入{self.colors['reset']}")
            
            # 語音生成模組
            self.voice_module = VoiceModule()
            print(f"{self.colors['system']}✓ 語音生成模組已載入{self.colors['reset']}")
            
            logger.info("所有AI模組初始化完成")
            print(f"{self.colors['system']}所有AI模組初始化完成！{self.colors['reset']}")
        except Exception as e:
            logger.error(f"初始化AI模組時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}初始化AI模組時發生錯誤: {str(e)}{self.colors['reset']}")
            raise
    
    def print_welcome(self):
        """
        顯示歡迎訊息
        """
        welcome_message = config.UI_CONFIG["cli"]["welcome_message"]
        print(f"\n{self.colors['bold']}{self.colors['system']}{welcome_message}{self.colors['reset']}\n")
        print(f"{self.colors['system']}輸入 'help' 查看可用命令{self.colors['reset']}\n")
    
    def print_help(self):
        """
        顯示幫助訊息
        """
        help_text = f"""
{self.colors['bold']}可用命令：{self.colors['reset']}
  {self.colors['bold']}基本命令：{self.colors['reset']}
    help                - 顯示此幫助訊息
    exit, quit          - 退出程式
    clear               - 清除對話歷史
  
  {self.colors['bold']}設定命令：{self.colors['reset']}
    thinking on/off     - 開啟/關閉思考過程顯示
    autoplay on/off     - 開啟/關閉自動播放語音
  
  {self.colors['bold']}語音命令：{self.colors['reset']}
    voice record        - 開始語音輸入 (需要麥克風)
    voice list          - 列出已保存的說話者
    voice select [名稱] - 選擇說話者
    voice clone [檔案] [名稱] - 從音頻檔案克隆語音
  
  {self.colors['bold']}文件命令：{self.colors['reset']}
    file add [檔案路徑] - 添加文件到知識庫
    file list           - 列出已添加的文件
    file clear          - 清除所有文件
        """
        print(help_text)
    
    def process_input(self, text_input: str):
        """
        處理文字輸入
        
        Args:
            text_input: 用戶輸入文字
        """
        # 檢查輸入是否為空
        if not text_input or text_input.strip() == "":
            print(f"{self.colors['error']}輸入不能為空{self.colors['reset']}")
            return
        
        try:
            # 記錄開始時間
            start_time = time.time()
            
            # 添加用戶消息到對話歷史
            user_message = {"role": "user", "content": text_input, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.conversation_history.append(user_message)
            
            # 分析輸入情緒
            emotion_distribution = self.emotion_module.analyze_input_emotion(text_input)
            main_emotion = max(emotion_distribution.items(), key=lambda x: x[1])
            
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
            response, thought_process = self.response_module.generate_response(
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
            
            # 計算處理時間
            end_time = time.time()
            process_time = end_time - start_time
            
            # 顯示思考過程
            if self.show_thinking:
                print(f"\n{self.colors['thinking']}思考過程：{self.colors['reset']}")
                for key, value in thought_process.items():
                    print(f"{self.colors['thinking']}{key}: {value}{self.colors['reset']}")
                print()
            
            # 顯示情緒分析
            print(f"{self.colors['system']}主要情緒: {main_emotion[0]} ({main_emotion[1]:.2f}){self.colors['reset']}")
            
            # 顯示回應
            print(f"{self.colors['assistant']}AI: {response}{self.colors['reset']}")
            
            # 顯示處理時間
            print(f"{self.colors['system']}處理時間: {process_time:.2f} 秒{self.colors['reset']}")
            
            # 播放語音
            if self.autoplay_audio:
                self._play_audio(audio_output_path)
            
        except Exception as e:
            logger.error(f"處理輸入時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}處理錯誤: {str(e)}{self.colors['reset']}")
    
    def process_command(self, command: str) -> bool:
        """
        處理命令
        
        Args:
            command: 用戶輸入的命令
            
        Returns:
            是否繼續運行
        """
        # 分割命令和參數
        parts = command.strip().split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []
        
        # 處理基本命令
        if cmd in ["exit", "quit"]:
            print(f"{self.colors['system']}再見！{self.colors['reset']}")
            return False
        elif cmd == "help":
            self.print_help()
        elif cmd == "clear":
            self.conversation_history = []
            print(f"{self.colors['system']}對話歷史已清除{self.colors['reset']}")
        
        # 處理設定命令
        elif cmd == "thinking":
            if len(args) > 0:
                if args[0].lower() == "on":
                    self.show_thinking = True
                    print(f"{self.colors['system']}已開啟思考過程顯示{self.colors['reset']}")
                elif args[0].lower() == "off":
                    self.show_thinking = False
                    print(f"{self.colors['system']}已關閉思考過程顯示{self.colors['reset']}")
                else:
                    print(f"{self.colors['error']}無效的參數，使用 'on' 或 'off'{self.colors['reset']}")
            else:
                print(f"{self.colors['error']}缺少參數，使用 'thinking on' 或 'thinking off'{self.colors['reset']}")
        
        elif cmd == "autoplay":
            if len(args) > 0:
                if args[0].lower() == "on":
                    self.autoplay_audio = True
                    print(f"{self.colors['system']}已開啟自動播放語音{self.colors['reset']}")
                elif args[0].lower() == "off":
                    self.autoplay_audio = False
                    print(f"{self.colors['system']}已關閉自動播放語音{self.colors['reset']}")
                else:
                    print(f"{self.colors['error']}無效的參數，使用 'on' 或 'off'{self.colors['reset']}")
            else:
                print(f"{self.colors['error']}缺少參數，使用 'autoplay on' 或 'autoplay off'{self.colors['reset']}")
        
        # 處理語音命令
        elif cmd == "voice":
            if len(args) > 0:
                if args[0] == "record":
                    self._record_voice()
                elif args[0] == "list":
                    self._list_speakers()
                elif args[0] == "select" and len(args) > 1:
                    self._select_speaker(args[1])
                elif args[0] == "clone" and len(args) > 2:
                    self._clone_voice(args[1], args[2])
                else:
                    print(f"{self.colors['error']}無效的語音命令{self.colors['reset']}")
            else:
                print(f"{self.colors['error']}缺少語音命令參數{self.colors['reset']}")
        
        # 處理文件命令
        elif cmd == "file":
            if len(args) > 0:
                if args[0] == "add" and len(args) > 1:
                    self._add_file(args[1])
                elif args[0] == "list":
                    self._list_files()
                elif args[0] == "clear":
                    self._clear_files()
                else:
                    print(f"{self.colors['error']}無效的文件命令{self.colors['reset']}")
            else:
                print(f"{self.colors['error']}缺少文件命令參數{self.colors['reset']}")
        
        # 處理一般輸入
        else:
            self.process_input(command)
        
        return True
    
    def _record_voice(self):
        """
        錄製語音並處理
        """
        try:
            print(f"{self.colors['system']}開始錄音 (5秒)...{self.colors['reset']}")
            
            # 錄製語音
            transcription = self.whisper_module.transcribe_from_microphone(duration=5)
            
            print(f"{self.colors['system']}錄音完成！{self.colors['reset']}")
            print(f"{self.colors['system']}轉錄文字: {transcription}{self.colors['reset']}")
            
            # 處理轉錄文字
            if transcription:
                self.process_input(transcription)
            else:
                print(f"{self.colors['error']}未能識別語音內容{self.colors['reset']}")
        except Exception as e:
            logger.error(f"錄製語音時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}錄製語音錯誤: {str(e)}{self.colors['reset']}")
    
    def _list_speakers(self):
        """
        列出已保存的說話者
        """
        try:
            speaker_dir = self.voice_module.speaker_embedding_path
            if not speaker_dir.exists():
                print(f"{self.colors['system']}尚未保存任何說話者{self.colors['reset']}")
                return
            
            speakers = []
            for file in speaker_dir.glob("*.npy"):
                speakers.append(file.stem)
            
            if speakers:
                print(f"{self.colors['system']}已保存的說話者：{self.colors['reset']}")
                for i, speaker in enumerate(speakers):
                    marker = "→" if speaker == self.current_speaker else " "
                    print(f"{self.colors['system']} {marker} {i+1}. {speaker}{self.colors['reset']}")
            else:
                print(f"{self.colors['system']}尚未保存任何說話者{self.colors['reset']}")
        except Exception as e:
            logger.error(f"列出說話者時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}列出說話者錯誤: {str(e)}{self.colors['reset']}")
    
    def _select_speaker(self, speaker_name: str):
        """
        選擇說話者
        
        Args:
            speaker_name: 說話者名稱
        """
        try:
            speaker_path = self.voice_module.speaker_embedding_path / f"{speaker_name}.npy"
            
            if not speaker_path.exists():
                print(f"{self.colors['error']}找不到說話者: {speaker_name}{self.colors['reset']}")
                return
            
            self.current_speaker = speaker_name
            print(f"{self.colors['system']}已選擇說話者: {speaker_name}{self.colors['reset']}")
        except Exception as e:
            logger.error(f"選擇說話者時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}選擇說話者錯誤: {str(e)}{self.colors['reset']}")
    
    def _clone_voice(self, audio_file: str, speaker_name: str):
        """
        從音頻文件克隆語音
        
        Args:
            audio_file: 音頻文件路徑
            speaker_name: 說話者名稱
        """
        try:
            audio_path = Path(audio_file)
            
            if not audio_path.exists():
                print(f"{self.colors['error']}找不到音頻文件: {audio_file}{self.colors['reset']}")
                return
            
            print(f"{self.colors['system']}正在從 {audio_file} 克隆語音...{self.colors['reset']}")
            
            # 提取說話者嵌入
            embedding = self.voice_module.extract_speaker_embedding(audio_path)
            
            if embedding is None:
                print(f"{self.colors['error']}提取語音特徵失敗{self.colors['reset']}")
                return
            
            # 保存說話者嵌入
            success = self.voice_module.save_speaker_embedding(speaker_name, embedding)
            
            if not success:
                print(f"{self.colors['error']}保存語音特徵失敗{self.colors['reset']}")
                return
            
            # 設置當前說話者
            self.current_speaker = speaker_name
            
            print(f"{self.colors['system']}成功克隆語音: {speaker_name}{self.colors['reset']}")
        except Exception as e:
            logger.error(f"克隆語音時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}克隆語音錯誤: {str(e)}{self.colors['reset']}")
    
    def _add_file(self, file_path: str):
        """
        添加文件到知識庫
        
        Args:
            file_path: 文件路徑
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                print(f"{self.colors['error']}找不到文件: {file_path}{self.colors['reset']}")
                return
            
            if path.suffix.lower() not in ['.txt', '.pdf']:
                print(f"{self.colors['error']}不支援的文件類型: {path.suffix}{self.colors['reset']}")
                return
            
            print(f"{self.colors['system']}正在處理文件: {file_path}{self.colors['reset']}")
            
            # 添加文件到RAG
            chunk_count = self.rag_module.add_file(path)
            
            print(f"{self.colors['system']}成功添加文件: {file_path}，分割為 {chunk_count} 個文本塊{self.colors['reset']}")
        except Exception as e:
            logger.error(f"添加文件時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}添加文件錯誤: {str(e)}{self.colors['reset']}")
    
    def _list_files(self):
        """
        列出已添加的文件
        """
        try:
            stats = self.rag_module.get_file_statistics()
            
            if not stats["files"]:
                print(f"{self.colors['system']}尚未添加任何文件{self.colors['reset']}")
                return
            
            print(f"{self.colors['system']}已添加的文件：{self.colors['reset']}")
            for i, (file_name, file_info) in enumerate(stats["files"].items()):
                print(f"{self.colors['system']} {i+1}. {file_name} ({file_info['file_type']}) - {file_info['chunks']} 個文本塊{self.colors['reset']}")
        except Exception as e:
            logger.error(f"列出文件時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}列出文件錯誤: {str(e)}{self.colors['reset']}")
    
    def _clear_files(self):
        """
        清除所有文件
        """
        try:
            # 清除RAG中的文件數據
            removed_count = self.rag_module.clear_file_data()
            
            print(f"{self.colors['system']}成功清除 {removed_count} 個文件塊{self.colors['reset']}")
        except Exception as e:
            logger.error(f"清除文件時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}清除文件錯誤: {str(e)}{self.colors['reset']}")
    
    def _play_audio(self, audio_path: Union[str, Path]):
        """
        播放音頻
        
        Args:
            audio_path: 音頻文件路徑
        """
        try:
            # 檢查是否存在播放器
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                os.system(f"afplay {audio_path}")
            elif system == "Linux":
                os.system(f"aplay {audio_path}")
            elif system == "Windows":
                os.system(f"start {audio_path}")
            else:
                print(f"{self.colors['system']}無法自動播放音頻，請手動播放: {audio_path}{self.colors['reset']}")
        except Exception as e:
            logger.error(f"播放音頻時發生錯誤: {str(e)}")
            print(f"{self.colors['error']}播放音頻錯誤: {str(e)}{self.colors['reset']}")
    
    def run(self):
        """
        運行命令列介面
        """
        self.print_welcome()
        
        running = True
        while running:
            try:
                # 獲取用戶輸入
                user_input = input(f"{self.colors['user']}{config.UI_CONFIG['cli']['prompt']}{self.colors['reset']}")
                
                # 處理命令
                running = self.process_command(user_input)
                
                # 添加空行
                print()
            except KeyboardInterrupt:
                print(f"\n{self.colors['system']}已中斷，輸入 'exit' 退出{self.colors['reset']}")
            except Exception as e:
                logger.error(f"處理命令時發生錯誤: {str(e)}")
                print(f"{self.colors['error']}錯誤: {str(e)}{self.colors['reset']}")

# 主函數
def main():
    # 創建並運行命令列介面
    interface = CLIInterface()
    interface.run()

if __name__ == "__main__":
    main()
