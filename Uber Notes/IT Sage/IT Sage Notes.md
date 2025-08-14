# Moveworks Meeting
Aug.13.2025

* Uber is going to build the plugins
* Moveworks going to call the plugins 
* Slack is the first UI Moveworks is supporting (notification and identification)
	* Custom chat apps (Slack)
* we can bring our mcp gateway / stratum to the existing framework

conversational agent 
* user feedback included
* action included 
* reference included - citation; **what about outdated information; if 2 articles are provided at the same time**
* 2 search endpoints (live search / stale search)
* session / conversational - dropping terms from the conversation 
* CANNOT DO images, spreadsheets

Question
1. reasoning engine, needs our own api (ml studio)
	1. 
	2. horizontal scalable 
	3. RL to train classifier
2. too many plugins provided, confusion? 
	1. ~~plugins contamination, if we want to restrict certain knowledge base to certain plugin, how can we do that;~~
	2. more clarification on the permission and access control
	3. search, do we need to provide all docs / just the search api
3. deployed within uber network / deployed outside on cloud
	1. not sure if uber will support 
4. multiple usecases, will it share the same service / need its own service
5. multi-turn user state; if its from slack, how is that handled 
6. what is the average cost, how many background context / cost