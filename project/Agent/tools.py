
from typing import Callable, Optional, Type,Union
from langchain.tools import Tool,tool,StructuredTool
from langchain_core.tools import BaseTool,BaseToolkit
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper,SerpAPIWrapper
from typing import Dict,Literal
from pydantic import BaseModel,Field
from langchain_tavily import TavilySearch
import requests,types
from langchain_community.agent_toolkits import JsonToolkit
import  trafilatura
tools:Dict[str, Union[BaseTool,dict[str,BaseTool]]] = {} #各个用户共享的tools列表
# 工具注册函数（支持函数或 BaseTool 实例）
AMAP_KEY="36785cb4214aae7d4ffaf78cbcdea94d"
OPEN_WEATHER_KEY="49887ecafdb17f6533764fc8f6001fe2"
TAVILY_KEY="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"
SERPAPI_KEY='5e2bff6c75babaec5cd988ff42de84d1301811c6bf74bc3b8223ee680240f8c1'
SERPER_KEY="354c335007f57fb4ee5cf2024d884bde966ee4b3"

def register_tool(func_tool: Union[Callable,BaseTool,BaseToolkit]=None,
                  name=None, description=None, args_schema: Optional[Type[BaseModel]] = None,
                  return_direct=False,user_tools=None):

    user_tools = tools if user_tools is None else user_tools
    if isinstance(func_tool, BaseTool):
        tool_name = func_tool.name
        user_tools[tool_name] = func_tool
    elif isinstance(func_tool,types.FunctionType) or isinstance(func_tool,types.MethodType):
        if name is None or description is None:
            raise ValueError("name 和 description 是必须的（对于函数形式）")
        tool = StructuredTool.from_function(
            func=func_tool,
            name=name,
            description=description,
            args_schema=args_schema,
            return_direct=return_direct
        )  ##注意agent默认将输入作为一个字符串，如果需要按指定字段传递，需要构建strutedTool；Tool装饰器底层也是调用该函数构建结构化工具
           ##这里制定了参数结构args_schema，如果还用Tool.from_function，会报参数过多的问题
        user_tools[name] = tool
    elif isinstance(func_tool, BaseToolkit):
        tool_classes=func_tool.get_tools()
        name=func_tool.__class__.__name__
        user_tools[name] = {}
        for tool_class in tool_classes:
            user_tools[func_tool.__class__.__name__][tool_class.name]=tool_class
    else:
        raise Warning(f"工具注册失败，func_tool 必须是 Callable , BaseTool,BaseToolkit 实例或者APIwrapper,提供的是{type(func_tool)}")


def tool_register(name: str, description: str, args_schema: Optional[Type[BaseModel]] = None, return_direct: bool = False):
    def decorator(func: Callable):
        register_tool(func_tool=func, name=name, description=description, args_schema=args_schema, return_direct=return_direct)
        return func
    return decorator


class WeatherInput(BaseModel):
    city:str=Field(description="城市名")
    mode:Literal["base","all"]=Field(description="决定获取当前天气还是预测天气."
                                                 "可选值：\"base\",表示获取当前天气"
                                     "\"all\",表示预测未来天气",default="base")

@tool_register(
    name="AMAP",
    description="用于获取某个地区当前的天气情况和未来3天的天气情况。输入可以是城市或者省份名称（如 '北京'）和 mode参数",
    args_schema=WeatherInput,
) #带参数的装饰器：先调用该函数，返回的函数才是真正的装饰器：@decorator: 因为装饰器只接受一个函数作为参数，如果要传入其余参数，需要再包一层函数用于接收参数：
#这里装饰器没有改变包装函数的行为，只是注册了工具
def get_weather(city: str,mode:str="base") -> str:
    """
    调用高德天气查询API，返回天气信息字符串
    """
    GAODE_API_KEY = AMAP_KEY
    url = f"https://restapi.amap.com/v3/weather/weatherInfo"

    try:
        if mode == "base":
            params = {
                "city": city,
                "key": GAODE_API_KEY,
                "extensions": "base",  # 当前天气；如果需要预报改为 all
                "output": "JSON"
            }
            response = requests.get(url, params=params)
            data = response.json()
            if data["status"] != "1":
                return f"❌ 查询失败: {data.get('info', '未知错误')}"

            weather_info = data["lives"][0]
            city_name = weather_info["city"]
            province=weather_info["province"]
            weather = weather_info["weather"]
            temperature = weather_info["temperature"]
            wind = weather_info["winddirection"]
            humidity = weather_info["humidity"]
            report_time = weather_info["reporttime"]
            ans=(
                f"📍 {city_name},{province} 当前天气,播报时间:{report_time}：\n"
            f"🌤️ 天气：{weather}\n"
            f"🌡️ 温度：{temperature}℃\n"
            f"💨 风向：{wind}\n"
            f"💧 湿度：{humidity}%\n"
            f"🕒 更新时间：{report_time}"
            )


        elif mode == "all":
            params = {
                "city": city,
                "key": GAODE_API_KEY,
                "extensions": "all",  # 当前天气；如果需要预报改为 all
                "output": "JSON"
            }
            response = requests.get(url, params=params)
            data = response.json()

            if data["status"] != "1":
                return f"❌ 查询失败: {data.get('info', '未知错误')}"

            data_info = data["forecasts"][0]

            city_name = data_info["city"]
            province = data_info["province"]
            report_time = data_info["reporttime"]
            weather_infos= data_info["casts"]
            ans= f"📍 {city_name},{province} 天气预报，预报时间:{report_time}：\n 今天是："

            for weather_info in weather_infos:

                date=weather_info["date"]
                week="日" if weather_info["week"]==7 else weather_info["week"]
                day_weather=weather_info["dayweather"]
                night_weather=weather_info["nightweather"]
                day_temp=weather_info["daytemp"]
                night_temp=weather_info["nighttemp"]
                day_wind=weather_info["daywind"]
                night_wind=weather_info["nightwind"]
                day_power=weather_info["daypower"]
                night_power=weather_info["nightpower"]

                ans+=(f"{date},星期{week}\n天气：{day_weather}转{night_weather},气温：{day_temp}-{night_temp}。白天风向：{day_wind},风力：{day_power}.夜晚风向"
                      f"{night_wind},风力{night_power}\n")
        else:
            raise ValueError(f"mode只允许是\"base\"或者\"all\",但获取的值是{mode}")

    except Exception as e:
        return f"❌ 请求异常：{str(e)}"
    return  ans

class url_args(BaseModel):
    url: str = Field(description="要爬取的网页URL")

@tool_register(
    name="fetch_url",
    description=(
        "用于爬取给定URL下的内容；"
        "如果搜索引擎结果不足以回答问题，可以使用该工具进一步抓取网页正文。"
    ),
    args_schema=url_args,
)
def fetch_url(url: str) -> str:
    """
    尝试提取网页正文，失败时返回提示信息。
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "❌ 无法下载网页内容，请检查URL是否有效。"

        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False
        )
        if not extracted:
            return "⚠️ 未能成功提取网页正文，可能是网页结构复杂或被反爬。"
        return extracted.strip()
    except Exception as e:
        return f"🚫 抓取网页时发生错误：{str(e)}"


#register_tool(DuckDuckGoSearchResults(name="DuckDuckGoSearch"))

register_tool(TavilySearch(tavily_api_key=TAVILY_KEY,name="TavilySearch")) #免费每月1000次

register_tool(GoogleSerperAPIWrapper(serper_api_key=SERPER_KEY).run,name="GoogleSerperSearch",
              description="谷歌搜索API，用来搜索问题")  #免费共计2500次？
register_tool(SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY).run,name="SerpAPIWrapper",description="搜索引擎，用来搜索问题")
#Free 250 searches / month
