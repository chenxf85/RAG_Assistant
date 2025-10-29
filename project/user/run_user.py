
import gradio as gr
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from user.log_in import login, logout, guest_login,loginToRegister
from user.sign_up import register
from user.MyBlocks import MyBlock,login_block, register_block, demo_app

# 创建 应用登录和注册页面，可以选择游客访问和用户访问（支持保存此前创建的数据库）
#gr.State是gradio中定义的一个可以在交互中持久化的变量（组件），可以用作输入输出组件，可以用变量本身的方式去访问。用于记录当前界面的状态，登录的用户，以及是否登录
#gr.update()是动态更新组件属性的方法。
#没有直接写整个run_user函数而是分了登录和注册block，是因为app界面如果要增加返回登录的按钮，需要以登录block作为输出组件（可见），隐藏app_block。
#而如果写成整体，run_user需要使用app_block，而app_block的布局由于返回登录按钮，所以需要以登录block作为输出组件，这样会出现问题。
#所以需要把登录block和登陆界面按钮进入app_block的逻辑分开写，这样app_block的布局就不会受到影响。
#代码复杂了一些，但是保证了模块的分离型。






def run_user(login_block:Myblock, register_block:Myblock,app_block:Myblock): #demo

        state = gr.State({"username": "", "logged_in": False})

        log_button2 = register_block["log_button2"]
        login_button1 = login_block["login_button1"]
        reg_button1 = login_block["reg_button1"]
        guest_button = login_block["guest_button"]
        reg_button2 = register_block["reg_button2"]
        logout_button = app_block["logout_button"]



        login_button1.click(
            login,
            inputs=[login_block["login_username"],login_block["login_password"] , state],
            outputs=[login_block["login_output"], state, login_block.block, app_block.block]
        ) #登录app

        log_button2.click(loginToRegister, inputs=[],
                          outputs=[register_block.block, login_block.block, login_block["login_output"]])  # 返回登录界面

        guest_button.click(guest_login, inputs=[state],
                           outputs=[login_block["login_output"]
                               , state, login_block.block, app_block.block]
                           )

        reg_button1.click(loginToRegister, inputs=[],
                          outputs=[login_block.block,register_block.block, login_block["login_output"]])   #登录跳转到注册界面
        reg_button2.click(register, inputs=[register_block["reg_username"],register_block["reg_password1"],register_block["reg_password2"] ],
                          outputs=[register_block["reg_output"], register_block.block, login_block.block, login_block["login_output"]])  #注册成功后跳转到登录界面
        logout_button.click(
            logout,
            inputs=[state],
            outputs=[state, app_block.block, login_block.block,login_block["login_output"]]
        )



#这是结合登录，注册和app_block使用的一个demo
if __name__ == "__main__":

    with gr.Blocks() as demo:

        login_block = login_block()
        register_block = register_block()
        app_block = demo_app()
        run_user(login_block, register_block,app_block)

    demo.launch()

