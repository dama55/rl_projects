import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torchvision.models as models
import numpy as np
from pathlib import Path
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

import matplotlib.pyplot as plt

import time, datetime

from abc import ABC, abstractmethod

from my_ppo.Util.FileUtils import exist_or_initialize_dir, add_to_file, write_to_file
from my_ppo.Util.TimeUtils import get_current_time_string

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class RLNet(nn.Module, ABC):
    """
    Abstract base class for reinforcement learning NetWork.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.device = device

    @abstractmethod
    def get_state_dict(self):
        """状態を定義するべきパラメータの辞書型を返す

        以下のように各モデルからパラメータを取り出して辞書として返す
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "channel_adapter": self.channel_adapter.state_dict()
        }
        
        """
        return {}

    @abstractmethod
    def set_state_dict(self, dict):
        """パラメータをセットする

        Args:
            dict (dict): メンバ変数を定義したパラメータ
        """
        # dict内の値を変数に設定
        self.__dict__.update(dict)

    @abstractmethod
    def get_parameter(self):
        """パラメータを取り出す
        これは学習可能なパラメータ情報のみを取り出す
        optimizerなどに渡す際に有用な情報を一括で取り出す

        Returns:
            array: 各パラメータを持つ辞書型の配列

        例
        [
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ]
        """
        return []


class AgentAbstract(ABC):
    """
    Abstract base class for reinforcement learning agents.
    """
    @abstractmethod
    def __init__(self,
                 state_dim,
                 action_dim, 
                 save_dir:Path = None, 
                 load_dir:Path = None,
                 save_every = None, #example 5e5
                 learn_every = 10 
                 ):
        #########################################
        # 環境設定
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ネットワークをセーブするステップ数
        self.save_every = save_every
        # 学習を実行するステップ数
        self.learn_every = learn_every

        #########################################
        # 静的な設定
        # デバイスのセット
        self.device = device
        
        # プログラムセーブ用のpath
        self.save_dir = save_dir
        if self.save_dir:
            self.log_path = self.save_dir / f"log"
        
        # プログラムロード用のpath
        self.load_dir = load_dir

        # エージェント初期化時刻保存
        self.initialized_time = get_current_time_string()
        #########################################
        # 動的な設定
        
        # ステップ数を管理．
        # ネットワークをloadして初期値を変更することもできる
        self.curr_step = 0

        

        # 保存をするかどうか
        self.save_log_or_not = True
        # 保存する最初のステップかどうか
        self.first_step = True

    def set_save_log(self, save_log_or_not):
        """ログを保存するかどうかをセットする
        """
        self.save_log_or_not = save_log_or_not

    def __initialize_file(self):
        """モデルセーブ用ディレクトリのログを初期化

        """
        # save_dirが指定してあるならログを初期化
        if self.save_dir:
            # ディレクトリの初期化
            exist_or_initialize_dir(self.save_dir)
            # 内容の定義
            content = \
            "==================================================\n" + \
            "Initialized new agent "+ self.initialized_time + "\n"
            # エージェント初期化情報をログに追加
            add_to_file(self.log_path, content)
		    
    @abstractmethod
    def act(self, state):
        """Select an action based on the current state.
        行動を出力する
        状態stateを元に
    

        Args:
            state (tensor): 状態

        Returns:
            tensor: 確定した行動を表すスカラー
        """
        pass

    @abstractmethod
    def update(self, state, next_state, action, reward, done):
        """Select an action based on the current state.
        行動を元に時状態が確定した時点での処理
        2024/12/30現段階の経験上やることは二つ
        1. 各ステップでの経験の蓄積, cache, add buffer, add memory
	        * オフポリシーやエピソードごとに学習するモデルなどで
		        各ステップのデータを蓄積する処理
        2. パラメータ更新 update, learn
            * 何ステップかごとに蓄積したデータを元に勾配の更新をするメソッド
        3. データのセーブ
            * 更新した際にパラメータをセーブする
    

        Args:
            state (tensor): 状態
            state (``LazyFrame``): 状態
		        next_state (``LazyFrame``): 次の状態
		        action (``int``): 行動
		        reward (``float``): 報酬
		        done(``bool``)): エピソードが終了したかどうか
		        
        Returns:
            numpy: lossなどの損失，Q値などの評価値を返す
        """
        pass

    @abstractmethod
    def get_state_dict(self):
        """保存の必要があるデータを辞書型として取り出す
        個別のアルゴリズムごとに追加で必要な変数があれば，
        このクラスを継承して，オーバーライドして新たにパラメータを追加する．

        Returns:
            dict: 取り出した辞書型
        """
        state_dict = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "save_every": self.save_every,
            "learn_every": self.learn_every,
            "curr_step": self.curr_step
        }
        return state_dict
    
    @abstractmethod
    def set_state_dict(self, dict):
        # dict内の値を変数に設定
        self.__dict__.update(dict)



    def save_model(self):
        """
        Save the agent's model to a file.
        エージェントのモデルを保存する
        動的，静的な変数も全てここで保存する
        """ 

        if self.save_dir and self.save_log_or_not:
            if self.first_step:
                # 初めに書き込むなら初期化する必要がある
                self.__initialize_file()
                self.first_step =  False

            save_path = self.save_dir / f"ppo_agent_{int(self.curr_step // self.save_every)}_eve_step_{self.save_every}.chkpt"
            torch.save(self.get_state_dict(), save_path)

            print(f"Saved checkpoint {int(self.curr_step // self.save_every)} to {save_path}")

            # ログへ情報を追加
            content = \
            f"saved checkpoint {int(self.curr_step // self.save_every):5.0f}_eve_step_{self.save_every:5.0f} "+get_current_time_string() + "\n"
            add_to_file(self.log_path, content=content)
        else: 
            print(f"Don't Saved checkpoint {int(self.curr_step // self.save_every)}")

    def load_model(self, num_chkpt:int):
        """
        Load a saved model from a file.
        エージェントのモデルをロードする
        
        動的，静的な変数も全てここでロードするのが理想
        """
        if self.load_dir:
            load_path = self.load_dir / f"ppo_agent_{num_chkpt}_eve_step_{self.save_every}.chkpt"
            dict = torch.load(load_path)
            # 取得した辞書型はset_state_dictで継承先のクラスごとに設定してもらう
            self.set_state_dict(dict)
            print(f"Loaded checkpoint {num_chkpt} from {load_path}")

            if self.save_dir and self.save_log_or_not:
                # ログに保存する状態かどうか
                if self.first_step:
                    # 初めに書き込むなら初期化する必要がある
                    self.__initialize_file()
                    self.first_step = False

                # ログへ情報を追加
                content = \
                f"loaded checkpoint {num_chkpt}_eve_step_{self.save_every} "+get_current_time_string() + "\n"
                add_to_file(self.log_path, content=content)

        else:
            print(f"Data hasn't be loaded because load_dir is None type")

class MetricAbstract(ABC):
    """
    Abstract base class for reinforcement learning Metric Rogger.
    """
    @abstractmethod
    def __init__(self, metric_name, noneable = False):
        """継承してオーバーライドしなおす必要あり
        メトリックの計算に必要なパラメータを設定

        Args:
            metric_name (_type_): _description_
        """
        ######################################
        # 環境依存のパラメータ
        ######################################
        # 静的パラメータ

        self.metric_name = metric_name

        # 毎ステップ存在する必要がないか
        # lossなどは毎ステップ計算されるわけではないのでnoneable=True
        self.noneable = noneable

        ######################################
        # 動的パラメータ
        # 各ステップごとのメトリックを保存する配列
        self.step_metrics = []
        # 各エピソードごとのメトリックを保存する配列
        self.episode_metrics = []
        # 複数エピソードのメトリック移動平均を保存する配列
        self.moving_ave_metrics = []

    

    def get_moving_metric(self):
        # 移動平均のデータ配列を返す
        return self.moving_ave_metrics
    
    def get_name(self):
        # メトリック名を返す
        return self.metric_name
    
    def log_step(self, **kwargs):
        # ステップ処理
        value =  self._calc_step_metric(**kwargs)
        if value != None:
            self.step_metrics.append(value)
    
    def log_episode(self):
        # エピソード処理
        value = self._calc_episode_metric()
        if value != None:
            self.episode_metrics.append(value)

    def log_moving_average(self):
        # 移動平均処理
        value = self._calc_moving_average()
        if value != None:
            self.moving_ave_metrics.append(value)

    
    def _extract_param(self, kwargs, param_key):
        """パラメータの辞書型から特定のパラメータを呼び出す関数

        Args:
            kwargs (dict): パラメータの辞書型
            param_key (string): パラメータの変数名

        Raises:
            KeyError: パラメータが辞書に存在しないときにエラー

        Returns:
            value: 取り出したパラメータを返す
        """
        if param_key not in kwargs:
            if self.noneable == True:
                # Noneでもいい場合はNoneを返す
                return None
            else:
                raise KeyError(f"Missing required parameter: '{param_key}'")
        return kwargs[param_key]
    
    @abstractmethod
    def get_moving_string(self):
        """ログファイルに出力するための移動平均を文字列として返す
        例えばこんな感じで直前の平均値を文字列とする
        f"{mean_ep_reward:15.3f}
        """
        raise NotImplementedError("Subclasses must not implement this method!")
        # 直線の記録を利用
        last_ave = self.moving_ave_metrics[-1]
        

    @abstractmethod
    def get_moving_title(self):
        """ログファイルのタイトルとして
        出力するためのタイトルを文字列として返す
        例えばこんな感じでタイトルの表記を決定する
        f"{'MeanLength':>15}
        """
        raise NotImplementedError("Subclasses must not implement this method!")



    @abstractmethod
    def _calc_step_metric(self, **kwargs):
        """各ステップごとにメトリックを実際に計算するクラス
        この抽象クラスを継承した先で各メトリクスごとにオーバーライドする

        Raises:
            NotImplementedError: この関数自体は呼び出してはいけない

        Returns:
            metric: 実際に計算されたメトリックの値
        """
        raise NotImplementedError("Subclasses must not implement this method!")

        #こんな感じでパラメータを取り出して適切な処理をして返す
        param = self._extract_param(kwargs, "param")
        if metric:#Noneではないことを確かめる
            # ここでパラメータを使ってメトリクスを計算
            metric = param
            # メトリックをそのまま返す
            return metric
    
    @abstractmethod
    def _calc_episode_metric(self):
        """各エピソードごとのメトリックを計算するクラス
        抽象クラスを継承した先でオーバーライド
        """
        raise NotImplementedError("Subclasses must not implement this method!")
        # ステップごとのメトリクスを利用して計算するのがいい
        episode_metric = np.mean(self.step_metrics)
        return episode_metric

    @abstractmethod
    def _calc_moving_average(self):
        """複数エピソード間でのメトリクスを計算するクラス
        抽象クラスを継承した先でオーバーライドする必要があるが
        この関数処理についてはほとんどが複数のエピソード間での移動平均を
        とることになると思われるため．super()._calc_moving_average()を呼び出すのでもあり

        Returns:
            _type_: _description_
        """
        # 複数のエピソード間での評価指標
        mean_ep_metric = np.round(np.mean(self.episode_metrics[-100:]), 3)
        return mean_ep_metric
    
    @abstractmethod
    def init_episode(self):
        """エピソードごとに初期化が必要なパラメータを初期化する
        """
        # メトリックの初期化
        self.step_metrics = []
        
    


    

        


class LoggerBase():
    """
    Abstract base class for reinforcement learning Logger.
    """
    def __init__(self, metrics,  save_dir: Path=None):
        """初期化に必要なコードは次の通り，
        * ログのセーブファイルを開いて，
          各メトリクスのカラム名を書き込む
        * 各メトリクスをプロットする用のグラフ名の作成
        * メトリック保存用の配列の用意と初期化
        * レコードタイムの開始

        Args:
            save_dir (str): ログをセーブする先のディレクトリ名
        """
        self.save_dir:Path = save_dir
        if save_dir:
            self.save_log = save_dir / "log"

        
        # メトリクスクラスを保持する配列
        self.metrics = metrics
        
        # Timing
        self.record_time = time.time()

        

        # 各ステップごとのステップ数を保存する配列
        self.step_steps = []
        # 各エピソードごとのステップ数を保存する配列
        self.episode_steps = []
        # 複数エピソードのステップ数移動平均を保存する配列
        self.moving_ave_steps = []

        # ログを保存するか
        self.save_log_or_not = True

        self.first_step = True

    def set_save_log(self, save_log_or_not):
        """ログを保存するかどうかをセットする
        """
        self.save_log_or_not = save_log_or_not
        

    def __initialize_file(self):
        """モデルセーブ用ディレクトリのログを初期化

        """
        
        # 保存用のディレクトリが決定している場合
        if self.save_log_or_not and self.save_dir:
            
            # ディレクトリの初期化
            exist_or_initialize_dir(self.save_dir)

            w_str = ""
            for metric in self.metrics:
                w_str += metric.get_moving_title()

            content = f"{'Episode':>8}{'Step':>8}" \
                    + w_str + \
                    "{'Time':>20}\n"

            # 新しくログに記述
            write_to_file(self.save_log, content)
    
    class RoggerBuilder():
        """ロガーに複数のメトリックを設定してビルドするためのクラス
        """
        def __init__(self, save_dir: Path=None):
            self.metrics = []
            self.save_dir = save_dir
        
        def add(self, metric):
            """メトリクスを追加

            Args:
                metric (class): メトリクスクラス
            """
            self.metrics.append(metric)
            return self
        
        def build(self):
            """メトリックを追加した後にビルドすることで状態を初期化する
            """
            

            return LoggerBase(save_dir=self.save_dir, metrics=self.metrics)

            


    
    
    def log_step(self, **kwargs):
        """各ステップでの処理を全てのメトリクスについて実行する
        """
        if self.first_step and self.save_log_or_not:
            # ログファイルの初期化
            self.__initialize_file()
            self.first_step = False

        for metric in self.metrics:
            metric.log_step(**kwargs)

        self.step_steps.append(kwargs["step"])



    def log_episode(self):
        """エピソードごとの処理を全てのメトリクスについて実行
        """
        for metric in self.metrics:
            metric.log_episode()
        
        self.episode_steps.append(self.step_steps[-1])
        self.init_episode()

    
    def init_episode(self):
        """エピソードに依存するデータの保存
        """
        for metric in self.metrics:
            metric.init_episode()
        # ステップ記録も削除
        self.step_steps = []

    def record(self, episode, step, **kwargs):
        moving_metrics =[]
        # グラフ作成用のメトリック時間移動平均の名前
        metric_names = []
        # 最新のメトリクス表示用のデータ
        mean_strings = []

        o_str = ""

        # レコード時のステップ数を記録
        self.moving_ave_steps.append(self.episode_steps[-1])
        
        for metric in self.metrics:
            # 各メトリックの移動平均を計算
            metric.log_moving_average()

            # メトリクスのエピソードごとの時間平均を取り出し
            name = metric.get_name()
            mv_metrics = metric.get_moving_metric()
            # コンソール出力用
            o_str += f"{name} {mv_metrics[-1]} - "
            
            # 移動平均
            moving_metrics.append(mv_metrics)
            # メトリックの名前
            metric_names.append(name)
            # ログ出力用の移動平均
            mean_strings.append(metric.get_moving_string())


        # 時間計算
        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)
        
        # コンソール出力
        print(
            f"Episode {episode} - "
            f"Step {step} - "
            +o_str+
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )
        if self.save_dir and self.save_log_or_not:
            # save_dirが存在し，保存する状態なら保存

            #データ文字列を結合
            # ログファイル出力
            w_str = "".join(mean_strings)

            content = f"{episode:8d}{step:8d}" \
                    +w_str+ \
                    f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            
            add_to_file(self.save_log, content)


            # グラフの作成
            for metric, metric_name in zip(moving_metrics, metric_names):
                plt.clf()
                plt.plot(self.moving_ave_steps, metric, label=f"moving_avg_{metric_name}")
                plt.legend()
                plt.savefig(self.save_dir / f"{metric_name}_plot", dpi=300, bbox_inches='tight')


        


    