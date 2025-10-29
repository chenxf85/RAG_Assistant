import gradio as gr
import json
import os
import traceback, logging

# 确保用户目录存在
os.makedirs(os.path.dirname(os.path.abspath("user/users.json")), exist_ok=True)

print("切换目录前的工作目录:", os.getcwd())  # 添加这行
#切换工作目录到项目文件夹
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("切换目录后的工作目录:", os.getcwd())  # 添加这行
USER_DB_FILE = "user/users.json"  # 用户信息保存的本地 JSON 文件

# 如果不存在用户数据库文件，初始化一个空字典并保存为 JSON 文件

#todo
# 注册需要增加一个禁止命名重复的函数
if not os.path.exists(USER_DB_FILE):
    with open(USER_DB_FILE, "w") as f:
        json.dump({}, f)


# 读取用户信息（返回字典 {用户名: 密码}）
def load_users():
    with open(USER_DB_FILE, "r") as f:
        return json.load(f)




# 保存新用户（写入 JSON 文件）
def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)


# 注册逻辑处理

def register(username, password1, password2):
    try:
        if not username or not password1 or not password2 :
            return "请填写所有信息！",gr.update(visible=True), gr.update(visible=False),""
        users = load_users()
        if username in users:
            return "用户名已存在！", gr.update(visible=True), gr.update(visible=False),""
        elif password1 != password2:
            return "两次密码不一致！", gr.update(visible=True), gr.update(visible=False),""
        save_user(username, password1)
        # 返回提示信息、隐藏注册界面、显示登录界面
        return f"注册成功！请切换到登录界面登录。", gr.update(visible=False), gr.update(visible=True),"" # 清空登录输入框
    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"注册发生错误：\n{error_info}")  # 错误输出到log.txt, 存在于项目根目录。
        print(error_info)
        return f"注册失败：{str(e)}",gr.update(visible=True), gr.update(visible=False),""





