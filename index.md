---
layout: page
title: K. Harding's E-Portfolio
---
Welcome to my e-portfolio for the University of Essex MSc in Artificial Intelligence.

<!--
## [Machine Learning](machine-learning.html)

{% assign ml_posts = site.posts | where_exp: "post", "post.course == nil" %}
<ul>
{% for post in ml_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
-->

## [Intelligence Agents](intelligent-agents.html)

{% assign ia_posts = site.posts | where: "course", "intelligent-agents" %}
{% if ia_posts.size > 0 %}
<ul>
{% for post in ia_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
_Coming soon._
{% endif %}
