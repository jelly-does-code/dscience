from facebook_scraper import *

set_user_agent(
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)")
for post in get_posts('nintendo'):
    print(post['text'][:50])