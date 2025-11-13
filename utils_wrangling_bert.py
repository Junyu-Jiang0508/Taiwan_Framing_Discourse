import pandas as pd
import numpy as np
import re
import hashlib
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import json
from datetime import datetime
import os

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("need to install: pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("need to install: pip install sentence-transformers scikit-learn")

try:
    from transformers import pipeline
except ImportError:
    print("need to install: pip install transformers torch")

class NLIMemo:
    """NLI结果缓存，避免重复计算"""
    def __init__(self, path="01_Data/02_processed_datasets/_nli_cache.jsonl"):
        self.path = path
        self.mem = {}
        # 确保目录存在
        pdir = os.path.dirname(self.path)
        if pdir:
            os.makedirs(pdir, exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        self.mem[j["key"]] = (j["margin"], j["status"])
                    except:
                        pass
    
    def get(self, key):
        return self.mem.get(key)
    
    def set(self, key, margin, status):
        self.mem[key] = (margin, status)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "margin": margin, "status": status}, ensure_ascii=False)+"\n")

def _hash256(s: str) -> str:
    """计算文本前256字符的MD5哈希"""
    return hashlib.md5(s[:256].encode("utf-8")).hexdigest()

# 强触发词（核心政治议题）
STRONG_KEYS = [
    # 两岸/身份/对外
    "兩岸","两岸","台海","主權","主权","國家認同","国家认同","一國兩制","一国两制","正名",
    # 国防/安全
    "國防","国防","兵役","軍演","飞弹","導彈","解放軍","海巡","潛艦國造","國艦國造",
    # 经济/能源
    "經濟","经济","通膨","物價","房價","核能","核四","再生能源","風電","光電","缺電","半導體","台積電",
    # 社会政策
    "居住正義","社宅","年金","健保","長照","托育","移工","原住民","性別平權","婚姻平權","同婚",
    # 选务/程序/国会
    "大選","總統","立法院","立委","政見","辯論","投票","不分區","三讀","質詢",
    # 国家符号/记忆政治
    "國旗","國歌","國徽","國慶","升旗","轉型正義","二二八","歷史正義","集體記憶","共同體","國格"
]

# "统一"多义词排除
ANTI_KEYS = ["統一發票","统一发票","統一超商","统一超商","統一企業","统一企业"]

def hit_strong_keys(s: str) -> bool:
    """检查是否命中强触发词（排除多义词）"""
    s = s or ""
    if any(a in s for a in ANTI_KEYS):
        return False
    return any(k in s for k in STRONG_KEYS)

class TextFilterStack:
    def __init__(self):
        self.whitelist_keywords = self._build_whitelist()
        self.blacklist_keywords = self._build_blacklist()
        self.bert_model = None
        self.nli_model = None
        
    def _build_whitelist(self) -> Set[str]:
        """
        覆盖：人名/党派（actor_terms）+ 制度/议题/机构/程序/两岸外交/安全/经济社会等（inst_terms）
        - 同时包含繁/简/英文/常用别称与口语写法
        - actor_terms 只放"人/党派/阵营"，其它进 inst_terms，便于"人名×体制/议题"共现判断
        """
        # 人名/党派/阵营（Actors）
        self.actor_terms = {
            # 2024 正副总统候选人 & 常见写法/英文
            '赖清德','賴清德','Lai Ching-te','William Lai','賴副總統','赖副总统',
            '萧美琴','蕭美琴','Bi-khim Hsiao','Hsiao Bi-khim',
            '侯友宜','Hou Yu-ih','侯市長','侯市长','新北市長','新北市长',
            '赵少康','趙少康','Jaw Shaw-kong',
            '柯文哲','Ko Wen-je','柯P','柯主席','臺北市長','台北市長','台北市长',
            '吴欣盈','吳欣盈','Hsin-ying Wu',
            
            # 主要政党（含繁简/英文缩写/正式全称/别称）
            '民进党','民進黨','民主進步黨','民主进步党','DPP',
            '国民党','國民黨','中國國民黨','中国国民党','KMT','Kuomintang',
            '民众党','民眾黨','台灣民眾黨','台湾民众党','TPP','Taiwan People\'s Party',
            
            # 其他常见政党/团体
            '時代力量','时代力量','NPP',
            '台灣基進','台湾基进','TSP','基進',
            '親民黨','亲民党','PFP',
            '新黨','新党',
            '綠黨','绿党',
            '社會民主黨','社会民主党',
            '無黨籍','无党籍',
            
            # 阵营/口语
            '綠營','绿营','藍營','蓝营','白營','白营','藍白','蓝白','藍白合','蓝白合'
        }
        
        # 创建小写版本用于英文匹配
        self.actor_terms_lower = {t.lower() for t in self.actor_terms}
        
        # 制度/议题/机构/程序（Institutions & Issues）
        self.inst_terms = {
            # 核心选举政治/程序
            '大選','大选','總統','总统','副總統','副总统','候選人','候选人',
            '政見','政见','政見發表會','政见发表会','辯論','辩论',
            '造勢','造势','掃街','扫街','拜票','競選','竞选','競總','竞总',
            '選區','选区','投開票所','投开票所','同意票','不同意票','廢票','废票',
            '賄選','賄选','買票','买票','補選','补选','罷免','罢免','罷免案','罢免案',
            '彈劾','弹劾','政黨輪替','政党轮替','民調','民调','支持度','黃金交叉','黄金交叉','死亡交叉',
            '立法院','立委','國會','国会','立院','不分區','不分区','區域立委','区域立委','席次',
            '表決','表决','協商','协商','朝野協商','院會','院会','臨時會','临时会',
            '質詢','质询','總質詢','总质询','黨團','党团','三讀','三读','草案','版本',
            
            # 宪政与法治/机构体系
            '憲法','宪法','修憲','修宪','憲政','宪政','違憲','违宪','釋憲','释宪','司法改革','司改','廉政','貪污','贪污',
            '中選會','中选会','中央選舉委員會',
            '行政院','司法院','監察院','监察院','考試院','考试院',
            '總統府','总统府','國安會','国安会','國防部','国防部','外交部','內政部','内政部','陸委會','陆委会','海基會','海基会',
            
            # 两岸/国家认同/对外
            '兩岸','两岸','主權','主权','九二共識','九二共识','台獨','台独','統一','统一','國家認同','国家认同',
            '一國兩制','一国两制','和平協議','和平协议','反滲透法','反渗透法','統戰','统一戰線',
            '認知作戰','认知作战','間諜','间谍','滲透','渗透','台海','台海危機',
            '邦交','斷交','断交','國際','国际','國際組織','国际组织','聯合國','联合国','WHA','APEC','CPTPP','RCEP','IPEF',
            '免簽','免签','簽證','签证','參與國際','参与国际','對外關係','对外关系','美台','美臺','美國國會','美国国会',
            
            # 国防/安全
            '國防','国防','國防自主','国防自主','軍購','军购',
            '兵役','徵兵','征兵','志願役','志愿役','教召',
            '軍演','军演','飛彈','飞弹','導彈','导弹','共機','共机','共軍','共军','解放軍','解放军',
            '海巡','海巡署','漁權','渔权','潛艦國造','潜舰国造','國艦國造','国舰国造',
            
            # 经济/产业/能源
            '經濟','经济','經濟成長','经济成长','民生經濟','民生经济','景氣','景气','投資','投资','外資','外资',
            '通膨','通货膨胀','物價','物价','油價','油价','房價','房价','囤房稅','囤房税',
            '稅制','税制','減稅','减税','稅收','税收','財政','财政','預算','预算',
            '產業政策','产业政策','供應鏈','供应链','出口','貿易','贸易',
            '半導體','半导体','晶片','芯片','台積電','台积电','護國神山','护国神山','國家隊','国家队',
            '能源','核能','核四','非核家園','非核家园','再生能源','風電','风电','離岸風電','离岸风电','光電','光电',
            '電價','电价','缺電','缺电','電網','电网','儲能','储能',
            
            # 社会政策/正义/劳动
            '居住正義','居住正义','社宅','社會住宅','社会住宅','都更',
            '年金改革','勞保','劳保','健保','長照','长照','托育','少子化',
            '勞工','劳工','基本工資','基本工资','工時','工时','移工','外籍看護','外籍看护',
            '性別平權','性别平权','婚姻平權','婚姻平权','同婚','平權','平权','原住民','偏鄉','偏乡','弱勢','弱势',
            
            # 教育/文化/历史/记忆（国家象征与叙事）
            '課綱','课纲','教改','學測','学测','學費','学费','校園安全','校园安全',
            '二二八','白色恐怖','戒嚴','戒严','解嚴','解严','轉型正義','转型正义','歷史正義','历史正义',
            '歷史敘事','历史叙事','集體記憶','集体记忆','共同記憶','共同记忆','創傷記憶','创伤记忆',
            '國旗','国旗','國歌','国歌','國徽','国徽','國號','国号','國格','国格','國慶','国庆','升旗','護照正名','护照正名',
            '正名','去中國化','去中国化','主體性','主体性','國族','国族','共同體','共同体',
            
            # 疫情/公共卫生
            '疫情','防疫','指揮中心','指挥中心','疫苗','高端','快篩','快筛','口罩',
            
            # 数位/媒体/平台治理
            '數位發展部','数字发展部','資安','资安','個資','个资',
            '數位中介服務法','数字中介服务法','中介法',
            '假訊息','假信息','假新聞','假新闻','網軍','网军','側翼','侧翼','帶風向','带风向',
            
            # 通用政策/治理用语
            '公共政策','政府','施政','施政報告','施政报告','議題','议题','民意','權利','权利','治安','能源轉型','碳費','碳费','淨零'
        }
        
        # 创建小写版本用于英文匹配
        self.inst_terms_lower = {t.lower() for t in self.inst_terms}
        
        return self.actor_terms | self.inst_terms
    
    def _build_blacklist(self) -> Set[str]:
        """
        黑名单只收"强商业/抽奖/导购/博彩"等明显无关/广告式触发词（繁/简/常见俗写），
        防止误删政策新闻。若你想更保守，可把娱乐/体育放到弱黑名单，用下游再判。
        """
        bl = {
            # 商业促销/导购/平台
            '包郵','包邮','免運','免运','秒殺','秒杀','下單','下单','到貨','到货','現貨','现货','預購','预购','上架',
            '返場','返场','清倉','清仓','折扣','特價','特价','促銷','促销','滿減','满减','直降','低至','送禮','送礼',
            '優惠','优惠','優惠券','优惠券','團購','团购','拼團','拼团','砍價','砍价','搶購','抢购','開團','开团',
            '開箱','开箱','種草','种草','拔草','晒单','種草清單','种草清单',
            '雙11','双11','雙十一','双十一','618','黑五','黑色星期五',
            '直播帶貨','直播带货','直播間','直播间','福利社','粉絲福利','粉丝福利','私訊下單','私信下单',
            '下單鏈接','下单链接','點我購買','点我购买','點擊搶購','点击抢购',
            '淘寶','淘宝','天貓','天猫','京東','京东','拼多多','蝦皮','虾皮','Shopee','MOMO','PChome','樂天','乐天',
            
            # 抽奖/彩票/博彩
            '抽獎','抽奖','轉發抽獎','转发抽奖','開獎','开奖','中獎','中奖','抽獎活動','抽奖活动',
            '彩票','体彩','福彩彩票','下注','博彩','賭博','赌博','賭場','赌场','德州撲克','德州扑克',
            
            # 金融骗局/币圈拉盘
            '幣圈','币圈','空投','薅羊毛','K線','K线','期貨','期货','合約','合约','炒幣','炒币',
            '穩賺','稳赚','快速回本','返利','代理招募','招代理','返傭','返佣',
            
            # 纯娱乐泛内容（酌情保留，避免误伤政策新闻）
            '明星八卦','八卦爆料','追星','飯圈','饭圈','應援','应援','專輯發售','专辑发售','演唱會','演唱会','影評','影评','劇評','剧评',
            '綜藝','综艺','偶像劇','偶像剧','片單','片单','票房',
            # 纯体育导向（若常做体育政策新闻，可移出黑名单）
            'NBA','CBA','MLB','中職','中职','球賽','球赛','比分','賽程','赛程','轉會','转会'
        }
        
        # 创建小写版本用于英文匹配
        self.blacklist_keywords_lower = {t.lower() for t in bl}
        
        return bl
    
    def preprocess_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def is_valid_length(self, text: str, min_len: int = 5, max_len: int = 1000) -> bool:
        return min_len <= len(text) <= max_len
    
    def get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def detect_near_duplicates(self, texts: List[str], threshold: float = 0.95) -> Set[int]:
        duplicates = set()
        text_hashes = {}
        
        for idx, text in enumerate(texts):
            text_hash = self.get_text_hash(text)
            
            if text_hash in text_hashes:
                duplicates.add(idx)
            else:
                text_hashes[text_hash] = idx
        
        return duplicates
    
    def hard_rule_filter(self, text: str) -> Tuple[bool, str]:
        text_lower = text.lower()
        
        # 黑名单先拦（中文原样 + 英文小写）
        for kw in self.blacklist_keywords | getattr(self, "blacklist_keywords_lower", set()):
            if kw in text or kw in text_lower:
                return False, f"blacklist: {kw}"
        
        # 白名单：人名×体制/议题共现（英文也要匹配小写）
        has_actor = any((kw in text) or (kw in text_lower) for kw in self.actor_terms | getattr(self, "actor_terms_lower", set()))
        has_inst = any((kw in text) or (kw in text_lower) for kw in self.inst_terms | getattr(self, "inst_terms_lower", set()))
        
        if has_actor and has_inst:
            return True, "cooccur: actor×institution"
        
        if (not has_actor) and has_inst:
            return True, "institution_only"
        
        return False, "not hit cooccur/inst"
    
    def taiwanicity_score(self, text: str) -> int:
        """计算台湾性得分（0-5分）"""
        score = 0
        if any(k in text for k in ['台灣','臺灣','中華民國']): 
            score += 1
        if any(k in text for k in ['立法院','行政院','中選會','總統府','國防部','外交部']):
            score += 1
        if any(k in text for k in ['總統','立委','政見','辯論','補選','罷免','三讀','表決']):
            score += 1
        if any(k in text for k in ['九二共識','台獨','統一','反滲透法','認知作戰','台海','邦交','斷交']):
            score += 1
        if any(k in text for k in ['台積電','半導體','能源','核能','電價','預算','稅制','社宅','年金','健保']):
            score += 1
        return score
    
    def _is_ambiguous_brand_or_sport(self, text: str) -> bool:
        """检测歧义词（品牌/体育，非政治）"""
        import re
        # 统一：棒球队、超商、企业、发票
        if re.search(r'統一(?=獅|發票|超商|企業)', text):
            return True
        # 国民：美食、旅游（非国民党）
        if re.search(r'國民(?=美食|旅遊)', text):
            return True
        return False
    
    def _policy_gate(self, text: str) -> Tuple[bool, str]:
        """台政闸门：强特征一票保留"""
        low = text.lower()
        
        # 台湾性得分≥2直接保留
        tw_score = self.taiwanicity_score(text)
        if tw_score >= 2 and not self._is_ambiguous_brand_or_sport(text):
            return True, f"gate: taiwanicity={tw_score}"
        
        # 命中Actor / Institution
        actor = any((kw in text) or (kw.lower() in low) 
                   for kw in self.actor_terms | getattr(self, "actor_terms_lower", set()))
        inst_hits = [kw for kw in self.inst_terms if (kw in text) or (kw.lower() in low)]
        inst_cnt = len(inst_hits)
        
        # 核心选举流程词
        core = {'總統','总统','候選人','候选人','選舉','选举','投票','政見','政见','辯論','辩论',
                '立法院','立委','中選會','行政院','院會','表決','不分區','席次','民調','造勢','競總'}
        has_core = any(c in text for c in core)
        
        # 判断规则
        if actor and inst_cnt > 0:
            return True, "gate: actor×inst"
        if has_core:
            return True, "gate: core_election"
        if inst_cnt >= 2:
            return True, f"gate: {inst_cnt} inst terms"
        
        # 强锚点
        strong_anchors = {
            '立法院','中選會','行政院','總統府','國防部','外交部','陸委會',
            '總統','候選人','政見','辯論','兩岸','主權','國防','外交','憲法',
            '九二共識','台獨','統一','一國兩制','反滲透法','認知作戰','介選',
            '邦交','斷交','聯合國','WHA','APEC','CPTPP',
            '台積電','半導體','供應鏈','能源','核能','電價','碳費','預算','稅制',
            '社宅','年金','健保','長照','托育','基本工資','兵役','教召','軍演','潛艦國造'
        }
        if any(a in text for a in strong_anchors):
            return True, "gate: strong_anchor"
        
        return False, ""
    
    def bm25_score(self, texts: List[str], query_keywords: List[str], top_k_ratio: float = 0.3) -> List[bool]:
        tokenized_corpus = [list(text) for text in texts]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 中文用字元级，去掉空格token
        tokenized_query = list("".join(query_keywords))
        scores = bm25.get_scores(tokenized_query)
        
        threshold_idx = int(len(scores) * (1 - top_k_ratio))
        threshold_score = sorted(scores)[threshold_idx] if len(scores) > 0 else 0
        
        return [score >= threshold_score for score in scores]
    
    def build_seeds(self, texts: List[str], k: int = 48) -> List[str]:
        """多桶种子：覆盖身份认同+选举流程+政策议题"""
        # 三个语义桶
        anchors_identity = [
            '国家认同','主权','两岸','统一','台独','中华民国','国格','国号','去中国化','反并吞','宪法','修宪',
            '一国两制','正名','护照正名','共同体','主体性','国族','集体记忆','历史叙事','二二八','白色恐怖','转型正义'
        ]
        anchors_politics = [
            '总统','副总统','候选人','政见','政见发表会','辩论','竞选','造势','拜票',
            '号次','抽签','立法院','立委','国会','不分区','席次','表决','协商','总质询','院会','临时会',
            '中选会','选区','投开票所','罢免','补选','政党轮替','民调','支持度','黄金交叉','死亡交叉'
        ]
        anchors_policy = [
            # 国防/对外
            '国防','军购','兵役','志愿役','军演','飞弹','共机','台海',
            '外交','邦交','断交','APEC','CPTPP','RCEP','WHA','美台','对外关系',
            # 经济/能源
            '经济','产业政策','供应链','出口','贸易','半导体','台积电','能源','核能','核四','再生能源','风电','电价','缺电',
            # 社会/居住/劳动
            '居住正义','社宅','年金','健保','长照','托育','基本工资','工时','移工','婚姻平权','原住民','偏乡','弱势',
            # 数位/治理/媒体
            '数位中介服务法','资安','个资','假新闻','网军','侧翼','带风向','能源转型','碳费','净零',
            # 公共卫生
            '防疫','指挥中心','疫苗','快筛','口罩'
        ]
        
        def pick(terms, m):
            hits = [t for t in texts if any(kw in t for kw in terms)]
            uniq = list(dict.fromkeys(hits))
            if not uniq:
                return []
            np.random.seed(42)
            return list(np.random.choice(uniq, size=min(m, len(uniq)), replace=False))
        
        k1, k2, k3 = int(0.40*k), int(0.35*k), int(0.25*k)
        seeds = pick(anchors_identity, k1) + pick(anchors_politics, k2) + pick(anchors_policy, k3)
        
        # 回填不足
        if len(seeds) < k:
            back = [t for t in texts if t not in seeds]
            back = list(dict.fromkeys(back))
            seeds += back[:max(0, k - len(seeds))]
        
        return seeds
    
    def load_bert_model(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        if self.bert_model is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                device = "cpu"
            self.bert_model = SentenceTransformer(model_name, device=device)
            print(f"  BERT model loaded on: {device}")
    
    def semantic_similarity_filter(
        self, 
        texts: List[str], 
        seed_texts: List[str],
        use_adaptive_threshold: bool = True
    ) -> Tuple[List[str], np.ndarray, float, float]:
        self.load_bert_model()
        
        # 批量更大、更快
        text_embeddings = self.bert_model.encode(texts, show_progress_bar=True, batch_size=128)
        seed_embeddings = self.bert_model.encode(seed_texts, show_progress_bar=True, batch_size=128)
        
        seed_center = np.mean(seed_embeddings, axis=0).reshape(1, -1)
        
        similarities = cosine_similarity(text_embeddings, seed_center).flatten()
        
        if use_adaptive_threshold:
            q05, q20 = np.quantile(similarities, [0.05, 0.20])
            keep_th = max(0.42, float(q20))  # 略放宽KEEP
            drop_th = min(0.28, float(q05))  # 略放宽DROP
        else:
            keep_th, drop_th = 0.42, 0.28
        
        print(f"  Adaptive thresholds: keep={keep_th:.3f}, drop={drop_th:.3f}")
        
        results = []
        for sim in similarities:
            if sim >= keep_th:
                results.append('KEEP')
            elif sim < drop_th:
                results.append('DROP')
            else:
                results.append('REVIEW')
        
        return results, similarities, float(keep_th), float(drop_th)
    
    def _truncate(self, text: str, max_chars: int = 256) -> str:
        return text[:max_chars]
    
    def load_nli_model(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        if self.nli_model is None:
            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except:
                device = -1
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.nli_model = pipeline("zero-shot-classification", model=model_name, device=device)
    
    def nli_filter(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[List[str], List[float]]:
        self.load_nli_model()
        
        total_texts = len(texts)
        print(f"  NLI filtering {total_texts} texts (two-stage optimization)...")
        
        # 阶段1：扩展正类覆盖L1/L2全域
        pos2 = [
            "台湾公共政策", "台湾政治", "台湾选举", "两岸关系", "国家认同",
            "竞选活动", "政见辩论", "造势动员", "选举流程",
            "居住正义", "劳工政策", "年金改革", "能源政策", "社会福利",
            "Taiwan politics", "Electoral politics", "Public policy", "Campaign"
        ]
        neg2 = ["娱乐", "体育", "商业促销", "抽奖导购", "生活服务"]
        labels2 = pos2 + neg2
        
        short_texts = [self._truncate(t, 256) for t in texts]
        
        print(f"    Stage 1: Binary classification...")
        out_fast = self.nli_model(
            short_texts,
            candidate_labels=labels2,
            hypothesis_template="这段文字主要与{}有关。",
            multi_label=True,
            batch_size=batch_size
        )
        
        if not isinstance(out_fast, list):
            out_fast = [out_fast]
        
        def margin_from(o):
            d = {lab: sc for lab, sc in zip(o["labels"], o["scores"])}
            return max(d.get(l, 0) for l in pos2) - max(d.get(l, 0) for l in neg2)
        
        margins = [margin_from(o) for o in out_fast]
        results = []
        need_calib_idx = []
        
        for i, m in enumerate(margins):
            if m >= 0.10:  # KEEP阈值
                results.append('KEEP')
            elif m <= 0.00:  # DROP阈值：仅负值才DROP
                results.append('DROP')
            else:
                results.append('REVIEW')
                need_calib_idx.append(i)
        
        print(f"    Stage 1 results: KEEP={results.count('KEEP')}, DROP={results.count('DROP')}, REVIEW={len(need_calib_idx)}")
        
        # 阶段2：仅对边界样本跑多标签校准
        if need_calib_idx:
            print(f"    Stage 2: Calibrating {len(need_calib_idx)} boundary samples...")
            
            pos_full = [
                "台湾公共政策", "台湾政治", "台湾选举", "两岸关系", "国家认同",
                "竞选活动", "政见辩论", "造势拜票", "选举流程", "竞选象征",
                "立法院与法案", "能源与环境政策", "社会福利与劳工", "教育与文化",
                "历史记忆与转型正义", "台湾国防安全", "台湾经济政策",
                "居住正义", "劳基法", "年金改革", "健保长照", "托育政策",
                "Taiwan politics", "Cross-strait relations", "Electoral campaign"
            ]
            neg_full = [
                "娱乐", "体育", "商业促销", "抽奖导购", "生活服务",
                "Entertainment", "Sports", "Commercial promotion"
            ]
            labels_full = pos_full + neg_full
            
            texts_calib = [short_texts[i] for i in need_calib_idx]
            out_full = self.nli_model(
                texts_calib,
                candidate_labels=labels_full,
                hypothesis_template="这段文字主要与{}有关。",
                multi_label=True,
                batch_size=batch_size
            )
            
            if not isinstance(out_full, list):
                out_full = [out_full]
            
            def margin_full(o):
                d = {lab: sc for lab, sc in zip(o["labels"], o["scores"])}
                return max(d.get(l, 0) for l in pos_full) - max(d.get(l, 0) for l in neg_full)
            
            for k, o in enumerate(out_full):
                m = margin_full(o)
                margins[need_calib_idx[k]] = m
                results[need_calib_idx[k]] = 'KEEP' if m >= 0.10 else ('DROP' if m <= 0.00 else 'REVIEW')
        
        print(f"    Final results: KEEP={results.count('KEEP')}, DROP={results.count('DROP')}, REVIEW={results.count('REVIEW')}")
        
        return results, margins
    
    def filter_pipeline(
        self,
        df: pd.DataFrame,
        text_column: str = 'sentence',
        seed_texts: List[str] = None,
        use_bm25: bool = True,
        use_semantic: bool = True,
        use_nli: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        stats = defaultdict(int)
        
        df = df.copy()
        df['preprocessed_text'] = df[text_column].apply(self.preprocess_text)
        
        valid_mask = df['preprocessed_text'].apply(
            lambda x: self.is_valid_length(x)
        )
        df = df[valid_mask].reset_index(drop=True)
        stats['after_length_filter'] = len(df)
        
        duplicates = self.detect_near_duplicates(df['preprocessed_text'].tolist())
        df = df[~df.index.isin(duplicates)].reset_index(drop=True)
        stats['after_deduplication'] = len(df)
        
        df['hard_rule_keep'] = False
        df['hard_rule_reason'] = ""
        
        for idx, text in enumerate(df['preprocessed_text']):
            keep, reason = self.hard_rule_filter(text)
            df.loc[idx, 'hard_rule_keep'] = keep
            df.loc[idx, 'hard_rule_reason'] = reason
        
        stats['after_hard_rule_keep'] = df['hard_rule_keep'].sum()
        
        candidate_df = df[df['hard_rule_keep']].copy()
        
        # 应用台政闸门（强特征一票保留）
        candidate_df['policy_gate'] = False
        candidate_df['policy_gate_reason'] = ""
        
        for idx in candidate_df.index:
            text = candidate_df.loc[idx, 'preprocessed_text']
            keep, why = self._policy_gate(text)
            candidate_df.loc[idx, 'policy_gate'] = keep
            candidate_df.loc[idx, 'policy_gate_reason'] = why
        
        stats['policy_gate_triggered'] = int(candidate_df['policy_gate'].sum())
        
        # 台湾性/强锚点旁路（priority_keep）- 直接设为KEEP跳过筛选
        candidate_df['priority_keep'] = False
        
        # 制度高置信词（强制度流程词）
        high_conf_inst = {'政見發表會','政见发表会','辯論會','辩论会','立法院','表決','表决',
                         '三讀','三读','院會','院会','朝野協商','朝野协商','投開票所','投开票所',
                         '不分區','不分区','席次','中選會','中选会','兩岸政策','两岸政策','國防','国防'}
        
        def is_high_priority(t):
            if self._is_ambiguous_brand_or_sport(t):
                return False
            # 台湾性≥2分
            if self.taiwanicity_score(t) >= 2:
                return True
            # 命中制度高置信词
            if any(h in t for h in high_conf_inst):
                return True
            return False
        
        priority_keep_mask = candidate_df['preprocessed_text'].apply(is_high_priority)
        candidate_df.loc[priority_keep_mask, 'priority_keep'] = True
        candidate_df.loc[priority_keep_mask, 'semantic_status'] = 'KEEP'
        candidate_df.loc[priority_keep_mask, 'semantic_similarity'] = 1.0
        candidate_df.loc[priority_keep_mask, 'nli_status'] = 'KEEP'
        candidate_df.loc[priority_keep_mask, 'nli_score'] = 1.0
        
        stats['priority_keep_count'] = int(priority_keep_mask.sum())
        print(f"  Priority keep: {stats['priority_keep_count']} high-confidence samples (bypassed filtering)")
        
        if use_semantic and len(candidate_df) > 0:
            # 只对未被priority_keep标记的样本进行语义筛选
            to_filter = candidate_df[~candidate_df.get('priority_keep', False)]
            
            if len(to_filter) > 0:
                if seed_texts is None:
                    seed_texts = self.build_seeds(candidate_df['preprocessed_text'].tolist(), k=48)
                    print(f"  Built {len(seed_texts)} seed texts from candidates")
                
                semantic_results, similarities, keep_th, drop_th = self.semantic_similarity_filter(
                    to_filter['preprocessed_text'].tolist(),
                    seed_texts,
                    use_adaptive_threshold=True
                )
                
                # 更新未被priority_keep的样本
                for idx, (result, sim) in zip(to_filter.index, zip(semantic_results, similarities)):
                    candidate_df.loc[idx, 'semantic_status'] = result
                    candidate_df.loc[idx, 'semantic_similarity'] = sim
            
            # 确保priority_keep样本保持KEEP状态
            if 'semantic_status' not in candidate_df.columns:
                candidate_df['semantic_status'] = 'KEEP'
                candidate_df['semantic_similarity'] = 1.0
            
            # 添加强触发词检测
            candidate_df['key_hit'] = candidate_df['preprocessed_text'].astype(str).apply(hit_strong_keys)
            
            stats['semantic_KEEP'] = (candidate_df['semantic_status'] == 'KEEP').sum()
            stats['semantic_DROP'] = (candidate_df['semantic_status'] == 'DROP').sum()
            stats['semantic_REVIEW'] = (candidate_df['semantic_status'] == 'REVIEW').sum()
            stats['key_hit_count'] = int(candidate_df['key_hit'].sum())
            
            # === RESCUE BLOCK: 对"语义DROP"的边缘样本做NLI救援 ===
            border = candidate_df[candidate_df['semantic_status']=='DROP'].copy()
            if len(border) > 0:
                top_n = min(200, max(1, int(0.05 * len(candidate_df))))
                border = border.sort_values('semantic_similarity', ascending=False).head(top_n)
                
                print(f"  Rescuing {len(border)} borderline DROP samples...")
                
                memo = NLIMemo()
                nli_texts, idx_map = [], []
                for idx, row in border.iterrows():
                    t = row['preprocessed_text']
                    k = _hash256(t)
                    hit = memo.get(k)
                    if hit is None:
                        nli_texts.append(self._truncate(t, 256))
                        idx_map.append((idx, k))
                    else:
                        m, st = hit
                        candidate_df.loc[idx, 'nli_status'] = st
                        candidate_df.loc[idx, 'nli_score'] = m
                
                if nli_texts:
                    rescue_res, rescue_scores = self.nli_filter(nli_texts, batch_size=32)
                    for (idx, k), st, m in zip(idx_map, rescue_res, rescue_scores):
                        candidate_df.loc[idx, 'nli_status'] = st
                        candidate_df.loc[idx, 'nli_score'] = m
                        memo.set(k, float(m), st)
                
                # 若NLI判为KEEP，则把语义DROP改成REVIEW
                rescued_idx = candidate_df[(candidate_df['semantic_status']=='DROP') & (candidate_df['nli_status']=='KEEP')].index
                if len(rescued_idx) > 0:
                    candidate_df.loc[rescued_idx, 'semantic_status'] = 'REVIEW'
                    print(f"  Rescued {len(rescued_idx)} samples from DROP to REVIEW")
                    stats['rescued_samples'] = int(len(rescued_idx))
        else:
            candidate_df['semantic_status'] = 'KEEP'
            candidate_df['semantic_similarity'] = 1.0
            candidate_df['key_hit'] = False
        
        if use_nli and len(candidate_df) > 0:
            # 只对semantic=REVIEW且未被硬保留的样本跑NLI
            # hard_rule_keep=True且semantic!=DROP的样本跳过NLI（省时且更稳）
            nli_candidate = candidate_df[
                candidate_df['semantic_status'].isin(['REVIEW']) &
                ~((candidate_df.get('hard_rule_keep', False)) & 
                  (candidate_df['semantic_status'] != 'DROP'))
            ].copy()
            
            if len(nli_candidate) > 0:
                # 使用缓存避免重复计算
                memo = NLIMemo()
                nli_texts, idx_map = [], []
                
                for idx, row in nli_candidate.iterrows():
                    text = row['preprocessed_text']
                    k = _hash256(text)
                    hit = memo.get(k)
                    
                    if hit is None:
                        nli_texts.append(text)
                        idx_map.append((idx, k))
                    else:
                        m, st = hit
                        candidate_df.loc[idx, 'nli_score'] = m
                        candidate_df.loc[idx, 'nli_status'] = st
                
                print(f"  NLI: {len(nli_texts)} new texts (cached: {len(nli_candidate) - len(nli_texts)})")
                
                if nli_texts:
                    nli_results, nli_scores = self.nli_filter(nli_texts, batch_size=32)
                    
                    for (idx, k), st, m in zip(idx_map, nli_results, nli_scores):
                        candidate_df.loc[idx, 'nli_status'] = st
                        candidate_df.loc[idx, 'nli_score'] = m
                        memo.set(k, float(m), st)
                
                # semantic=KEEP的样本自动标记为NLI KEEP
                semantic_keep = candidate_df[
                    (candidate_df['semantic_status'] == 'KEEP') &
                    (candidate_df['nli_status'].isna())
                ]
                candidate_df.loc[semantic_keep.index, 'nli_status'] = 'KEEP'
                candidate_df.loc[semantic_keep.index, 'nli_score'] = 1.0
                
                # hard_rule_keep=True且未进NLI的样本也标记为KEEP
                hard_skipped = candidate_df[
                    (candidate_df.get('hard_rule_keep', False)) &
                    (candidate_df['semantic_status'] == 'REVIEW') &
                    (candidate_df['nli_status'].isna())
                ]
                candidate_df.loc[hard_skipped.index, 'nli_status'] = 'KEEP'
                candidate_df.loc[hard_skipped.index, 'nli_score'] = 1.0
                
                stats['nli_KEEP'] = (candidate_df['nli_status'] == 'KEEP').sum()
                stats['nli_DROP'] = (candidate_df['nli_status'] == 'DROP').sum()
                stats['nli_REVIEW'] = (candidate_df['nli_status'] == 'REVIEW').sum()
                stats['nli_cached'] = len(nli_candidate) - len(nli_texts) if nli_texts else 0
                stats['nli_computed'] = len(nli_texts)
                stats['nli_skipped_hard_rule'] = len(hard_skipped)
        else:
            candidate_df['nli_status'] = 'KEEP'
            candidate_df['nli_score'] = 1.0
        
        def final_decision(row):
            # 0) 黑名单一票否决
            if 'blacklist' in str(row.get('hard_rule_reason', '')):
                return 'DROP'
            
            # 1) actor×inst共现：强制保留（不得DROP）
            if row.get('policy_gate_reason') == 'gate: actor×inst':
                # NLI强烈反对才降为REVIEW，否则KEEP
                if row.get('nli_status') == 'DROP' and row.get('nli_score', 0) <= -0.15:
                    return 'REVIEW'
                return 'KEEP'
            
            # 2) priority_keep直通
            if row.get('priority_keep', False):
                return 'KEEP'
            
            # 3) policy_gate其他原因（core_election/strong_anchor等）
            if row.get('policy_gate', False):
                return 'KEEP'
            
            # 4) institution_only：允许通过语义/NLI判断，但至少REVIEW
            if 'institution_only' in str(row.get('hard_rule_reason', '')):
                if row.get('semantic_status') == 'KEEP' and row.get('nli_status') == 'KEEP':
                    return 'KEEP'
                return 'REVIEW'  # 保护制度类文本，至少进REVIEW
            
            # 5) NLI=KEEP 直接保留
            if row.get('nli_status') == 'KEEP':
                return 'KEEP'
            
            # 6) semantic=DROP，但命中强触发词则降级为REVIEW保护
            if row.get('semantic_status') == 'DROP':
                if row.get('key_hit', False):
                    return 'REVIEW'
                return 'DROP'
            
            # 7) NLI=DROP 仅在margin明确负向时才丢弃
            if row.get('nli_status') == 'DROP':
                m = row.get('nli_score', -1.0)
                # 如果margin > 0 或命中强触发词，降级为REVIEW
                if m > 0.00 or row.get('key_hit', False):
                    return 'REVIEW'
                return 'DROP'
            
            # 8) semantic=KEEP直接KEEP
            if row.get('semantic_status') == 'KEEP':
                return 'KEEP'
            
            # 9) 其余REVIEW
            return 'REVIEW'
        
        candidate_df['final_status'] = candidate_df.apply(final_decision, axis=1)
        
        stats['final_KEEP'] = (candidate_df['final_status'] == 'KEEP').sum()
        stats['final_DROP'] = (candidate_df['final_status'] == 'DROP').sum()
        stats['final_REVIEW'] = (candidate_df['final_status'] == 'REVIEW').sum()
        
        self._quality_check(candidate_df)
        self._sanity_check(candidate_df, stats)
        
        return candidate_df, dict(stats)
    
    def _quality_check(self, df: pd.DataFrame):
        print("\n  [Quality Check] Sample outputs:")
        
        drop_sample = df[df['final_status'] == 'DROP']
        if len(drop_sample) > 0:
            sample_size = min(3, len(drop_sample))
            samples = drop_sample.sample(sample_size, random_state=42)
            print(f"\n  DROP samples ({sample_size}/{len(drop_sample)}):")
            for idx, row in samples.iterrows():
                text = row['preprocessed_text'][:80]
                sim = row.get('semantic_similarity', 0)
                print(f"    - [{sim:.3f}] {text}...")
        
        review_sample = df[df['final_status'] == 'REVIEW']
        if len(review_sample) > 0:
            sample_size = min(3, len(review_sample))
            samples = review_sample.sample(sample_size, random_state=42)
            print(f"\n  REVIEW samples ({sample_size}/{len(review_sample)}):")
            for idx, row in samples.iterrows():
                text = row['preprocessed_text'][:80]
                sim = row.get('semantic_similarity', 0)
                print(f"    - [{sim:.3f}] {text}...")
    
    def _sanity_check(self, df: pd.DataFrame, stats: Dict):
        """统计policy_gate效果和回流情况"""
        print("\n  [Sanity Check]:")
        
        if 'priority_keep' in df.columns:
            pri_keep = df[df.get('priority_keep', False)]
            print(f"    Priority keep (taiwanicity≥2 or high-conf inst): {len(pri_keep)} samples")
        
        if 'policy_gate_reason' in df.columns:
            # actor×inst共现保护
            actor_inst = df[df['policy_gate_reason'] == 'gate: actor×inst']
            print(f"    Actor×Inst protected: {len(actor_inst)} samples (forced KEEP)")
            
            # 统计如果没有保护会被误删的
            would_drop = actor_inst[
                (actor_inst['semantic_status'] == 'DROP') | 
                (actor_inst['nli_status'] == 'DROP')
            ]
            if len(would_drop) > 0:
                print(f"      - Rescued from DROP: {len(would_drop)} samples")
        
        if 'policy_gate' in df.columns:
            gate_keep = df[df['policy_gate'] & (df['final_status'] == 'KEEP')]
            print(f"    Policy gate total: {len(gate_keep)} / {df['policy_gate'].sum()}")
        
        # institution_only保护统计
        inst_only = df[df.get('hard_rule_reason', '').str.contains('institution_only', na=False)]
        if len(inst_only) > 0:
            inst_review = inst_only[inst_only['final_status'] == 'REVIEW']
            print(f"    Institution_only: {len(inst_only)} total, {len(inst_review)} → REVIEW")
        
        if 'key_hit' in df.columns:
            key_protect = df[df['key_hit'] & (df['final_status'].isin(['KEEP', 'REVIEW']))]
            print(f"    Strong keys保护: {len(key_protect)} samples")
        
        if 'rescued_samples' in stats and stats['rescued_samples'] > 0:
            print(f"    NLI救援: {stats['rescued_samples']} samples")
        
        # 误删风险检测
        potential_fn = df[
            (df['final_status'] == 'DROP') &
            (df.get('key_hit', False) | df.get('policy_gate', False))
        ]
        if len(potential_fn) > 0:
            print(f"    ⚠ WARNING: {len(potential_fn)} potential false negatives!")
            print(f"      (DROP but hit strong features - please review)")

def process_single_file(
    input_file: str, 
    output_file: str, 
    text_column: str,
    filter_stack: TextFilterStack,
    save_all_categories: bool = True
) -> Dict:
    df = pd.read_csv(input_file)
    
    result_df, stats = filter_stack.filter_pipeline(
        df,
        text_column=text_column,
        use_bm25=False,
        use_semantic=True,
        use_nli=True
    )
    
    keep_df = result_df[result_df['final_status'] == 'KEEP']
    
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存KEEP样本到主输出文件
    keep_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # 保存DROP和REVIEW样本到单独文件
    if save_all_categories:
        base_name = os.path.splitext(output_file)[0]
        
        drop_df = result_df[result_df['final_status'] == 'DROP']
        if len(drop_df) > 0:
            drop_file = base_name + '_DROP.csv'
            drop_df.to_csv(drop_file, index=False, encoding='utf-8-sig')
        
        review_df = result_df[result_df['final_status'] == 'REVIEW']
        if len(review_df) > 0:
            review_file = base_name + '_REVIEW.csv'
            review_df.to_csv(review_file, index=False, encoding='utf-8-sig')
    
    stats['input_file'] = input_file
    stats['output_file'] = output_file
    stats['original_count'] = len(df)
    
    return stats

def batch_process_datasets():
    import os
    import glob
    import time
    
    filter_stack = TextFilterStack()
    
    all_stats = []
    
    news_input_dir = '01_Data/01_raw_datasets/01_news_datasets'
    news_output_dir = '01_Data/02_processed_datasets/01_news_datasets'
    
    news_files = glob.glob(os.path.join(news_input_dir, '*.csv'))
    
    x_input_dir = '01_Data/01_raw_datasets/03_X_datasets'
    x_output_dir = '01_Data/02_processed_datasets/03_X_datasets'
    
    x_files = glob.glob(os.path.join(x_input_dir, '*.csv'))
    
    total_files = len(news_files) + len(x_files)
    processed_files = 0
    
    print(f"Total files to process: {total_files}")
    print(f"  News files: {len(news_files)}")
    print(f"  X files: {len(x_files)}")
    print("-" * 60)
    
    start_time = time.time()
    file_times = []
    
    for input_file in news_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(news_output_dir, filename)
        
        file_start = time.time()
        print(f"\nProcessing [{processed_files + 1}/{total_files}]: {filename}")
        
        try:
            stats = process_single_file(
                input_file=input_file,
                output_file=output_file,
                text_column='sentence',
                filter_stack=filter_stack
            )
            
            file_time = time.time() - file_start
            file_times.append(file_time)
            processed_files += 1
            
            all_stats.append(stats)
            
            avg_time = sum(file_times) / len(file_times)
            remaining_files = total_files - processed_files
            estimated_remaining = avg_time * remaining_files
            
            hours = int(estimated_remaining // 3600)
            minutes = int((estimated_remaining % 3600) // 60)
            seconds = int(estimated_remaining % 60)
            
            print(f"Completed: {filename}")
            print(f"  Original: {stats['original_count']}")
            print(f"  KEEP: {stats.get('final_KEEP', 0)}")
            print(f"  DROP: {stats.get('final_DROP', 0)}")
            print(f"  REVIEW: {stats.get('final_REVIEW', 0)}")
            print(f"  Processing time: {file_time:.1f}s")
            print(f"Progress: {processed_files}/{total_files} files")
            
            if hours > 0:
                print(f"Estimated remaining time: {hours}h {minutes}m {seconds}s")
            elif minutes > 0:
                print(f"Estimated remaining time: {minutes}m {seconds}s")
            else:
                print(f"Estimated remaining time: {seconds}s")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            processed_files += 1
    
    for input_file in x_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(x_output_dir, filename)
        
        file_start = time.time()
        print(f"\nProcessing [{processed_files + 1}/{total_files}]: {filename}")
        
        try:
            df_temp = pd.read_csv(input_file)
            
            if 'Tweet Content' in df_temp.columns:
                text_col = 'Tweet Content'
            elif 'sentence' in df_temp.columns:
                text_col = 'sentence'
            else:
                print(f"  Skipped: no valid text column found")
                processed_files += 1
                continue
            
            stats = process_single_file(
                input_file=input_file,
                output_file=output_file,
                text_column=text_col,
                filter_stack=filter_stack
            )
            
            file_time = time.time() - file_start
            file_times.append(file_time)
            processed_files += 1
            
            all_stats.append(stats)
            
            avg_time = sum(file_times) / len(file_times)
            remaining_files = total_files - processed_files
            estimated_remaining = avg_time * remaining_files
            
            hours = int(estimated_remaining // 3600)
            minutes = int((estimated_remaining % 3600) // 60)
            seconds = int(estimated_remaining % 60)
            
            print(f"Completed: {filename}")
            print(f"  Original: {stats['original_count']}")
            print(f"  KEEP: {stats.get('final_KEEP', 0)}")
            print(f"  DROP: {stats.get('final_DROP', 0)}")
            print(f"  REVIEW: {stats.get('final_REVIEW', 0)}")
            print(f"  Processing time: {file_time:.1f}s")
            print(f"Progress: {processed_files}/{total_files} files")
            
            if hours > 0:
                print(f"Estimated remaining time: {hours}h {minutes}m {seconds}s")
            elif minutes > 0:
                print(f"Estimated remaining time: {minutes}m {seconds}s")
            else:
                print(f"Estimated remaining time: {seconds}s")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            processed_files += 1
    
    stats_output = '01_Data/02_processed_datasets/00_batch_stats.json'
    os.makedirs(os.path.dirname(stats_output), exist_ok=True)
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    all_stats_serializable = convert_to_serializable(all_stats)
    
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(all_stats_serializable, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"Total files processed: {len(all_stats)}/{total_files}")
    
    if hours > 0:
        print(f"Total processing time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Total processing time: {minutes}m {seconds}s")
    else:
        print(f"Total processing time: {seconds}s")
    
    if len(file_times) > 0:
        avg_time = sum(file_times) / len(file_times)
        print(f"Average time per file: {avg_time:.1f}s")
    
    total_original = sum(s.get('original_count', 0) for s in all_stats)
    total_keep = sum(s.get('final_KEEP', 0) for s in all_stats)
    total_drop = sum(s.get('final_DROP', 0) for s in all_stats)
    total_review = sum(s.get('final_REVIEW', 0) for s in all_stats)
    
    total_after_hard = sum(s.get('after_hard_rule_keep', 0) for s in all_stats)
    total_filtered = total_keep + total_drop + total_review
    total_hard_dropped = total_original - total_after_hard
    
    print(f"\nOverall statistics:")
    print(f"  Total original rows: {total_original}")
    print(f"\n  After preprocessing & hard rules:")
    print(f"    - Passed hard rules: {total_after_hard} ({total_after_hard/total_original*100:.1f}%)")
    print(f"    - Dropped by hard rules: {total_hard_dropped} ({total_hard_dropped/total_original*100:.1f}%)")
    print(f"\n  After semantic & NLI filtering:")
    print(f"    - KEEP: {total_keep} ({total_keep/total_after_hard*100:.1f}% of candidates, {total_keep/total_original*100:.1f}% of original)")
    print(f"    - DROP: {total_drop} ({total_drop/total_after_hard*100:.1f}% of candidates, {total_drop/total_original*100:.1f}% of original)")
    print(f"    - REVIEW: {total_review} ({total_review/total_after_hard*100:.1f}% of candidates, {total_review/total_original*100:.1f}% of original)")
    print(f"\n  Final retention rate: {total_keep/total_original*100:.1f}%")
    print(f"\nStatistics saved to: {stats_output}")
    print("=" * 60)

def main():
    batch_process_datasets()

if __name__ == "__main__":
    main()

