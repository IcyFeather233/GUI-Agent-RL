prompt_score_system = """作为一名GUI和强化学习领域的专家，你将接收到某个任务的完整截图和交互的文本描述。你需要评估任务链中某个步骤的价值，类似于强化学习中的价值函数。以下是详细的标准和解释。

## 输入内容说明：
1. 任务：当前GUI任务的简要描述，例如在Android GUI中实现“获取香港酒店价格”任务。
2. 任务的完整操作描述及对应的截图序列
   (1) 操作文本描述：包含11种GUI操作。具体字段及其含义如下：
      [1] DUAL_POINT：双击屏幕上的特定位置。如果是链接或软件，则会进入；如果是文本，则会被选中。“click_point”用一个二维数组表示点击的位置，相对于截图的左上角，范围从0.0到1.0。
         - 示例: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
      [2] TYPE：发送文本的操作类型。注意，此操作仅发送文本，不会执行任何点击以聚焦元素或按下回车键提交文本。
         - 示例: "action_type": "TYPE", "typed_text": "capital of England"
      [3] PRESS_BACK：返回上一页。通常是上一个网页。
         - 示例: "action_type": "PRESS_BACK"
      [4] PRESS_HOME：返回系统主页。当当前屏幕不是所需屏幕时，使用此操作返回主屏幕，以便重新选择需要进入的程序。
         - 示例: "action_type": "PRESS_HOME"
      [5] PRESS_ENTER：按下回车键以执行步骤。通常在确认输入文本后，使用此操作开始搜索。
         - 示例: "action_type": "PRESS_ENTER"
      [6] STATUS_TASK_COMPLETE：用于指示所需任务已完成并重置环境的操作。如果任务已经完成且无需进一步操作，也应使用此操作。例如，任务是在Wi-Fi已开启时再次开启Wi-Fi。
         - 示例: "action_type": "STATUS_TASK_COMPLETE"
      [7] STATUS_TASK_IMPOSSIBLE：用于指示所需任务无法完成并重置环境的操作。这可能是由于UI变化、Android版本差异等原因。
         - 示例: "action_type": "STATUS_TASK_IMPOSSIBLE"
      [8] SCROLL_DOWN：向下滚动。
         - 示例: "action_type": "SCROLL_DOWN"
      [9] SCROLL_UP：向上滚动。
         - 示例: "action_type": "SCROLL_UP"
      [10] SCROLL_LEFT：向左滚动。
         - 示例: "action_type": "SCROLL_LEFT"
      [11] SCROLL_RIGHT：向右滚动。
         - 示例: "action_type": "SCROLL_RIGHT"
   (2) 每次操作前对应的截图。如果操作类型为“DUAL_POINT”，则会在图像中用红点标记点击位置。    
3. 当前待评估的操作及对应的截图。

## 评估标准：
以下是两个级别的详细描述。需要注意基于当前截图所采取的操作是否促进了任务的高效执行，而不是当前截图显示的内容与任务的相关性：
   级别1：该操作不是当前完成任务的最佳选择，可能会导致任务流程的偏离。例如：
      (1) 输入了错误的文本。
      (2) 点击了可能导致广告的按钮。
      (3) 在任务未实际完成时宣布任务成功。
   级别2：该操作是当前完成任务的最佳且正确的选择。例如：
      (1) 当显示任务完成时，显示的内容可以完全实现任务。
      (2) 当进入不相关的界面时，可以通过执行“PRESS_HOME”返回主屏幕。
      (3) 选择最正确的入口点以完成当前任务。

## 输出要求：
- 格式: {"rating": int, "explanation": str}。不要包含此格式之外的任何额外字符
- “rating”字段应为1或2，表示评估级别。“explanation”字段应解释导致此评级的评估过程，不包含当前步骤之后的操作描述（未来的操作被视为未知）。

## 示例输入：
任务要求：英国的首都是哪里？
操作及截图：
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
step 2: "action_type": "PRESS_ENTER"
step 3: "action_type": "STATUS_TASK_COMPLETE"
当前操作：
step 2: "action_type": "PRESS_ENTER"

## 示例输出：
{"rating": 2, "explanation": "在输入‘capital of England’后按下回车键是获取任务要求答案的适当步骤，这是实现任务目标的最佳操作。"}

"""

prompt_score_user = """任务要求: {}
操作及截图: {}
当前操作: 
{}
"""

prompt_critic_system = """作为一名GUI和强化学习领域的专家，你将接收到某个任务的历史交互的文本描述。你需要评估当前操作，类似于强化学习中的价值函数。以下是详细的标准和解释。

## 输入内容说明：
1. 任务：当前GUI任务的简要描述，例如在Android GUI中实现“获取香港酒店价格”任务。
2. 历史操作描述
   包含11种GUI操作。具体字段及其含义如下：
   [1] DUAL_POINT：双击屏幕上的特定位置。如果是链接或软件，则会进入；如果是文本，则会被选中。“click_point”用一个二维数组表示点击的位置，相对于截图的左上角，范围从0.0到1.0。
      - 示例: "action_type": "DUAL_POINT", "click_point": [0.5, 0.5]
   [2] TYPE：发送文本的操作类型。注意，此操作仅发送文本，不会执行任何点击以聚焦元素或按下回车键提交文本。
      - 示例: "action_type": "TYPE", "typed_text": "capital of England"
   [3] PRESS_BACK：返回上一页。通常是上一个网页。
      - 示例: "action_type": "PRESS_BACK"
   [4] PRESS_HOME：返回系统主页。当当前屏幕不是所需屏幕时，使用此操作返回主屏幕，以便重新选择需要进入的程序。
      - 示例: "action_type": "PRESS_HOME"
   [5] PRESS_ENTER：按下回车键以执行步骤。通常在确认输入文本后，使用此操作开始搜索。
      - 示例: "action_type": "PRESS_ENTER"
   [6] STATUS_TASK_COMPLETE：用于指示所需任务已完成并重置环境的操作。如果任务已经完成且无需进一步操作，也应使用此操作。例如，任务是在Wi-Fi已开启时再次开启Wi-Fi。
      - 示例: "action_type": "STATUS_TASK_COMPLETE"
   [7] STATUS_TASK_IMPOSSIBLE：用于指示所需任务无法完成并重置环境的操作。这可能是由于UI变化、Android版本差异等原因。
      - 示例: "action_type": "STATUS_TASK_IMPOSSIBLE"
   [8] SCROLL_DOWN：向下滚动。
      - 示例: "action_type": "SCROLL_DOWN"
   [9] SCROLL_UP：向上滚动。
      - 示例: "action_type": "SCROLL_UP"
   [10] SCROLL_LEFT：向左滚动。
      - 示例: "action_type": "SCROLL_LEFT"
   [11] SCROLL_RIGHT：向右滚动。
      - 示例: "action_type": "SCROLL_RIGHT"
3. 当前待评估的操作及对应的截图（每次操作前的截图。如果操作类型为“DUAL_POINT”，则会在图像中用红点标记点击位置。）

## 评估标准：
以下是两个级别的详细描述。需要注意基于当前截图所采取的操作是否促进了任务的高效执行，而不是当前截图显示的内容与任务的相关性：
   级别1：该操作不是当前完成任务的最佳选择，可能会导致任务流程的偏离。例如：
      (1) 输入了错误的文本。
      (2) 点击了可能导致广告的按钮。
      (3) 在任务未实际完成时宣布任务成功。
   级别2：该操作是当前完成任务的最佳且正确的选择。例如：
      (1) 当显示任务完成时，显示的内容可以完全实现任务。
      (2) 当进入不相关的界面时，可以通过执行“PRESS_HOME”返回主屏幕。
      (3) 选择最正确的入口点以完成当前任务。

## 输出要求：1或2 (整数)

## 示例输入：
任务要求：英国的首都是哪里？
历史操作：
step 0: "action_type": "DUAL_POINT", "click_point": "[0.524, 0.06]"
step 1: "action_type": "TYPE", "typed_text": "capital of England"
当前操作及截图：
step 2: "action_type": "PRESS_ENTER"

## 示例输出：
2

"""

prompt_critic_user = """任务要求: {}
历史操作: 
{}
当前操作及截图: 
<image>
{}
"""

prompt_general = """你是一名评估截图是否成功完成任务的专家。

=====示例=====
截图: <image>
任务: 打开设置。
问: 如果我打开了设置，我应该在截图中看到什么？
答: 我应该看到我在设置应用中。截图显示的是移动设备的主屏幕，显示了各种应用图标，包括设置应用图标，但设置应用并未打开。
状态: 失败

截图: <image>
任务: 查找华盛顿特区的酒店
问: 如果我搜索了华盛顿特区的酒店，我应该在截图中看到什么？
答: 我应该看到我在华盛顿特区酒店的搜索结果页面。截图显示的是Google搜索页面，搜索字段中填充了查询“hotels in washington dc”，并显示了与华盛顿特区酒店相关的搜索建议，但并未显示任何华盛顿特区酒店的搜索结果。
状态: 失败

截图: <image>
任务: 波特兰有什么好餐厅？
问: 如果我搜索了波特兰的好餐厅，我应该在截图中看到什么？
答: 我应该看到我在波特兰好餐厅的搜索结果页面。截图显示的是Google搜索页面，搜索输入字段为“good restaurant in portland”，并显示了地图结果预览，显示了波特兰附近的商业位置，如“Li Pigeon”、“Portland City Grill”和“Higgins”。
状态: 成功

截图: <image>
任务: In-N-Out的菜单上有什么？
问: 如果我搜索了In-N-Out的菜单，我应该在截图中看到什么？
答: 我应该看到In-N-Out的菜单页面，包括产品名称、缩略图和价格。截图显示的是Google搜索页面，搜索输入字段为“In-N-Out menu”，并显示了一些In-N-Out的页面片段，表示潜在的菜单项，但并未显示实际的菜单。
状态: 失败

截图: <image>
任务: 苏里南有什么新闻？
问: 如果我搜索了苏里南的新闻，我应该在截图中看到什么？
答: 我应该看到一些苏里南的新闻，例如某人在苏里南做了某事或发生了某些事故。截图显示的是Google搜索页面，搜索输入字段为“Suriname news today”，并显示了一些表示潜在新闻项的页面片段，但并未显示实际的新闻。
状态: 失败

截图: <image>
任务: 芝加哥的天气如何？
问: 如果我搜索了芝加哥的天气，我应该在截图中看到什么？
答: 我应该看到一些确切的值，如温度、湿度、风速和芝加哥的天气状况。截图显示的是Google搜索页面，搜索输入字段为“weather in Chicago”，并显示了一些表示潜在天气信息的页面片段。尽管其中一个页面片段包含一些天气信息，但信息不够全面，无法确定芝加哥的天气。
状态: 失败

截图: <image>
任务: 设置一个下午6点的闹钟。
问: 如果我设置了一个下午6点的闹钟，我应该在截图中看到什么？
答: 我应该看到时钟应用中的一些闹钟，包括一个下午6点的闹钟已激活。截图显示的是在时钟应用中尝试设置下午6点的闹钟，但闹钟尚未设置。
状态: 失败

截图: <image>
任务: 今天法国的新闻是什么？
问: 如果我搜索了今天法国的新闻，我应该在截图中看到什么？
答: 我应该看到一些今天法国的新闻，例如某人在法国做了某事或发生了某些事故。截图显示我在france24.com网站上，但被一个cookie同意横幅挡住。
状态: 失败

截图: <image>
任务: 今天法国的新闻是什么？
问: 如果我搜索了今天法国的新闻，我应该在截图中看到什么？
答: 我应该看到一些今天法国的新闻，例如某人在法国做了某事或发生了某些事故。截图显示我在france24.com网站上，可以看到新闻，例如关于奥运火炬的新闻。
状态: 成功

=====你的回合=====
截图: <image>
任务: {}
请按以下格式回答：
问: 如果我<重复任务>，我应该在截图中看到什么？
答: 我应该首先看到<预期内容>，然后在给定的截图中看到<实际内容>。
状态: 成功或失败（不要返回其他内容）
以“问:”开头。
"""

prompt_webshop = """你是一名评估截图是否成功完成任务的专家。

=====示例=====
截图: <image>
任务: 访问bestbuy.com
问: 如果我访问了bestbuy.com，我应该在截图中看到什么？
答: 我应该看到我在Best Buy网站上，通常显示Best Buy的标志和一些特色产品和类别。截图显示我在Google搜索中搜索“bestbuy.com”（并显示了一些搜索建议），而不是在Best Buy网站上。
状态: 失败

截图: <image>
任务: 访问costco.com
问: 如果我访问了costco.com，我应该在截图中看到什么？
答: 我应该看到我在Costco网站上，通常显示主页和一些特色产品和类别。截图显示我在Costco网站上，显示了一些特色产品和类别。
状态: 成功

截图: <image>
任务: 访问bestbuy.com，搜索“macbook”
问: 如果我访问了bestbuy.com并搜索了“macbook”，我应该在截图中看到什么？
答: 我应该看到我在Best Buy网站上，并显示“macbook”的搜索结果。截图显示我在Best Buy网站上，并显示了“macbook”的几个搜索建议，但并未显示产品的搜索结果，通常包括价格和产品详细信息。
状态: 失败

截图: <image>
任务: 访问ebay.com，搜索“corsair k70”
问: 如果我访问了ebay.com并搜索了“corsair k70”，我应该在截图中看到什么？
答: 我应该看到我在eBay网站上，并显示“corsair k70”的搜索结果。截图显示我在eBay网站上，并显示了“corsair k70”的一些搜索建议，但并未显示产品的搜索结果，通常包括价格和产品详细信息。
状态: 失败

截图: <image>
任务: 访问walmart.com，搜索“macbook air”
问: 如果我访问了walmart.com并搜索了“macbook air”，我应该在截图中看到什么？
答: 我应该看到我在Walmart网站上，并显示“razer huntsman”的搜索结果。截图显示我在Google搜索中，并显示了“macbook air”的一些搜索建议，而不是在Walmart网站上。
状态: 失败

截图: <image>
任务: 访问walmart.com，搜索“razer huntsman”
问: 如果我访问了walmart.com并搜索了“razer huntsman”，我应该在截图中看到什么？
答: 我应该看到我在Walmart网站上，并显示“razer huntsman”的搜索结果。截图显示我在Walmart网站上，但并未显示“razer huntsman”的搜索结果，通常包括产品详细信息和价格。
状态: 失败

截图: <image>
任务: 访问ebay.com，搜索“lenovo thinkpad”
问: 如果我访问了ebay.com并搜索了“lenovo thinkpad”，我应该在截图中看到什么？
答: 我应该看到我在eBay网站上，并显示“lenovo thinkpad”的搜索结果。截图显示我在eBay网站上，并显示了“lenovo thinkpad”的几个搜索结果。
状态: 成功

截图: <image>
任务: 访问ebay.com，搜索“razer thresher”，选择第一个条目
问: 如果我访问了ebay.com并进入了“razer thresher”搜索结果的第一个条目，我应该在截图中看到什么？
答: 我应该看到我在eBay网站上，并显示razer thresher产品的详细信息，如产品的大图、价格和产品详细信息。截图显示我在eBay网站上，但显示了多个“razer thresher”的搜索结果，这意味着用户尚未选择搜索结果的第一个条目。
状态: 失败

截图: <image>
任务: 访问target.com，搜索“razer kraken”，并选择第一个条目
问: 如果我访问了target.com并进入了“razer kraken”搜索结果的第一个条目，我应该在截图中看到什么？
答: 我应该看到我在Target网站上，并显示razer thresher产品的详细信息，如产品的大图、价格和产品详细信息。截图显示我在Google搜索中，而不是在Target网站上。
状态: 失败

截图: <image>
任务: 访问ebay.com，搜索“acer predator”，并选择第一个条目
问: 如果我访问了ebay.com并进入了“acer predator”搜索结果的第一个条目，我应该在截图中看到什么？
答: 我应该看到我在eBay网站上，并显示acer predator产品的详细信息，如产品的大图、价格和产品详细信息。截图显示我在eBay网站上，但显示了多个“acer predator”的搜索结果，这意味着用户尚未选择搜索结果的第一个条目。
状态: 失败

截图: <image>
任务: 访问bestbuy.com，搜索“macbook”，选择第一个条目
问: 如果我访问了bestbuy.com并进入了“macbook”搜索结果的第一个条目，我应该在截图中看到什么？
答: 我应该看到我在eBay网站上，并显示macbook产品的详细信息，如产品的大图、价格和产品详细信息。截图显示我在eBay网站上，并显示了Macbook Air的详细信息，包括价格和产品详细信息。
状态: 成功

=====你的回合=====
截图: <image>
任务: {}
请按以下格式回答：
问: 如果我<重复任务>，我应该在截图中看到什么？
答: 我应该首先看到<预期内容>，然后在给定的截图中看到<实际内容>。
状态: 成功或失败（不要返回其他内容）
以“问:”开头。
"""

def build_prompt_general(config, task, image_path):
   if "general" in config["output_name"]:
      image_list = [
         "data/images/screenshot_menu.png", 
         "data/images/screenshot_hotel.png", 
         "data/images/screenshot_restaurant.png",
         "data/images/screenshot_foodmenu.png", 
         "data/images/screenshot_news.png", 
         "data/images/screenshot_weather.png", 
         "data/images/screenshot_alarm.png", 
         "data/images/screenshot_frenchnews_blocked.png", 
         "data/images/screenshot_frenchnews_okay.png",
         image_path
      ]
      
      return prompt_general.format(task).split("<image>"), image_list
   else:
      image_list = [
         "data/images/step1_bestbuy.png",
         "data/images/step1_costco.png",
         "data/images/step2_bestbuy.png",
         "data/images/step2_ebay.png",
         "data/images/step2_walmart.png",
         "data/images/step2_walmart2.png",
         "data/images/step2_ebay2.png",
         "data/images/step3_ebay.png",
         "data/images/step3_target.png",
         "data/images/step3_ebay2.png",
         "data/images/step3_bestbuy.png",
         image_path
      ]    
      
      return prompt_webshop.format(task).split("<image>"), image_list