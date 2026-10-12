---
layout: page
title: Machine Learning
permalink: /machine-learning.html
---

Overview of content from Machine Learning 2024 A.

{% assign ml_posts = site.posts | where_exp: "post", "post.course == nil" %}
<ul>
{% for post in ml_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
