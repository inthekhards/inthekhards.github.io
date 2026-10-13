---
layout: page
title: Intelligence Agents
permalink: /intelligent-agents.html
---

Overview of content from the Intelligence Agents module.

{% assign ia_posts = site.posts | where: "course", "intelligent-agents" %}
{% if ia_posts.size > 0 %}
<ul>
{% for post in ia_posts %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
{% endfor %}
</ul>
{% else %}
_Projects coming soon._
{% endif %}
