import gradio as gr
class MyBlock():

   # 自定义类，是为了可以以字典的形式访问布局内部组件（这些组件在不同block之间需要传递）
    def __init__(self, block:gr.Group,*args,**kwargs):
        self.block=block

        self._component_registry = {}

        for (name, component) in args:

            self.register(name, component)
            # 也支持关键字形式
        for name,component in kwargs.items():
            self.register(name, component)
        #
        # for (name, component,default) in args:
        #
        #     self.register(name, component,default)
        #     # 也支持关键字形式
        # for name, (component,default) in kwargs.items():
        #     self.register(name, component, default)
    def register(self, name, component):
        #获取变量名称

        self._component_registry[name] = component
        return component

    def __getitem__(self, name):
        return self._component_registry[name]

    def reset(self ):
        #这种方式实际上不是很严谨，理论上gradio组件的值的更新需要通过回调函数来修改，否则可能出现异步更新问题；  这种更新方式在DB.value.create_db_info 直接回调会出错？
        #下面这种方式实际上对于复杂组件，不好恢复到初始值，比如gr.Slider初始化参数过多
        #可以传入函数参数是一个{gr.component,{arg_default_value:str}}
        #这里先不具体实现,可以参考knowledgeDB.reset在login时调用
        for (component,default) in self._component_registry.values():
            component.value=default
        pass

def theme_block(scale=1):

    js_code = """
        () => {
            const dark_mode_on = document.body.classList.toggle('dark');
            return [dark_mode_on ? "浅色模式" : "深色模式"];
        }
        """
    theme_button = gr.Button("深色模式", elem_id="dark-mode-button", variant="primary", scale=scale)
    theme_button.click(fn=None, inputs=[], outputs=[theme_button], js=js_code)


LOGO_PATH = "../figures/logo.png"

def login_block() ->MyBlock:  #在block之间传递信息

        # 初始化状态，记录登录状态和当前用户名;

        # 登录界面（默认可见）
        with gr.Column(visible=True) as login_block:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    theme_block(scale=1)
                    gr.Row(scale=1)

                gr.Column(scale=2)
                gr.Image(value=LOGO_PATH, height=150,scale=2, min_width=1, show_label=False, show_download_button=False,
                         container=False)  #
                 # 为了使图片出现在合适的位置，用column占位
                gr.Column(scale=4)
            with gr.Column(scale=1):
                login_username = gr.Textbox(label="用户名")
                login_password = gr.Textbox(label="密码", type="password")
                login_button1 = gr.Button("登录")
                reg_button1= gr.Button("注册")
                guest_button =gr.Button("游客访问")
                login_output = gr.Textbox(label="提示")

            gr.Markdown("""
                提醒：<br>
               1. 用户登录可以管理知识库文件，并且自定义api_key <br>
               2. 支持游客快速访问，但是知识库文件不单独维护，且只支持使用给定的key <br>                      
            
               """)
            return MyBlock(login_block,
                            ("login_username",login_username),
                            ("login_password",login_password),
                            ("login_button1",login_button1),
                            ("reg_button1",reg_button1),
                            ("guest_button",guest_button),
                            ("login_output",login_output))

def register_block()->MyBlock:
        # 注册界面（默认不可见）

        with gr.Column(visible=False) as register_block:   #Group 是 Blocks 中的一个布局元素，它将子项分组在一起，以便它们之间没有任何填充或边距。
            with gr.Row(equal_height=True):
                theme_block()
                gr.Column(scale=2)
                gr.Markdown("<h1><center>注册<center></h1>")
                gr.Column(scale=3)
            reg_username = gr.Textbox(label="用户名")
            reg_password1 = gr.Textbox(label="密码", type="password")
            reg_password2 = gr.Textbox(label="确认密码", type="password")
            reg_button2 = gr.Button("注册")
            log_button2= gr.Button("返回登录")
            reg_output = gr.Textbox(label="提示")

        return MyBlock(register_block,
                        ("reg_username",reg_username),
                        ("reg_password1",reg_password1),
                        ("reg_password2",reg_password2),
                        ("reg_button2",reg_button2),
                        ("log_button2",log_button2),
                        ("reg_output",reg_output))
            # 登录按钮绑定登录函数，登录后根据返回值控制界面显示

def demo_app():
    with gr.Column(visible=False) as app_block:
        welcome_text = gr.Markdown("欢迎使用应用！")
        app_info = gr.Textbox(label="这里是你的应用内容")  # 可以放任何应用内容
        logout_button = gr.Button("退出登录")
        # 退出登录按钮绑定逻辑，退出后返回到登录和注册界面

        # 登录按钮绑定登录函数，登录后根据返回值控制界面显示

    return MyBlock(app_block,("logout_button",logout_button))