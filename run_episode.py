from oilenv import OilDeliveryEnv

# ファイルパスを指定して環境をインスタンス化
file_path = "D:\OneDrive\oilopt\plan.xlsx"
env = OilDeliveryEnv(file_path, inbound_reward_coef=1.0, outbound_reward_coef=1.0, negative_content_penalty=1.0)

# 環境をテスト
env.reset()
fixed_action = [0] * env.action_space.shape[0]  # アクションを初期化
fixed_action[0] = 1  # タンク0に油種Aを1単位（100）入れる
for _ in range(10):
    action = env.action_space.sample()  # ランダムなアクションをサンプリング
    next_state, reward, done, info = env.step(fixed_action)
    if done:
        break

