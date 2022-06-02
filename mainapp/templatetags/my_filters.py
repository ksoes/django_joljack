from django import template

register = template.Library()

@register.filter(name='getlength')
def getlength(num):  # get num's length and return range(length)
    return range(len(num))

@register.filter()
def get_at_index(object_list, index):
    return object_list[index]

