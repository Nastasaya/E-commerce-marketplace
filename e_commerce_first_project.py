#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
from datetime import date,timedelta


# Существует некий маркетплейс. Необходимо проанализировать работу приложения в части взаимодействия с клиентами. Для начала создадим датафреймы с полученными данными.

# In[2]:


customer_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-poplavskaja-20/Промежуточный проект/olist_customers_dataset.csv')
orders_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-poplavskaja-20/Промежуточный проект/olist_orders_dataset.csv', 
            parse_dates=['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date'])
items_df = pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-poplavskaja-20/Промежуточный проект/olist_order_items_dataset.csv', parse_dates=['shipping_limit_date'])


# In[3]:


# customers_datase.csv — таблица с уникальными идентификаторами пользователей
# customer_id — позаказный идентификатор пользователя (колонка для merge)
# customer_unique_id —  уникальный идентификатор пользователя (аналог паспорта)

# orders_dataset.csv —  таблица заказов
# customer_id —  позаказный идентификатор пользователя (колонка для merge)
# order_id —  уникальный идентификатор заказа (чек)
# order_status —  статус заказа
# order_purchase_timestamp —  время создания заказа
# order_approved_at        —  время подтверждения оплаты заказа
# order_delivered_carrier_date —  время передачи заказа в логистическую службу
# order_delivered_customer_date —  время доставки заказа
# order_estimated_delivery_date —  обещанная дата доставки

# order_items_dataset.csv —  товарные позиции, входящие в заказы
# order_id —  уникальный идентификатор заказа (номер чека)
# order_item_id —  идентификатор товара внутри одного заказа
# product_id —  ид товара (аналог штрихкода)
# shipping_limit_date —  максимальная дата доставки продавцом для передачи заказа партнеру по логистике

# Уникальные статусы заказов в таблице orders_dataset:
# created —  создан
# approved —  подтверждён
# invoiced —  выставлен счёт
# processing —  в процессе сборки заказа
# shipped —  отгружен со склада
# delivered —  доставлен пользователю
# unavailable —  недоступен
# canceled —  отменён


# In[4]:


customer = customer_df.drop(['customer_zip_code_prefix', 'customer_city', 'customer_state'], axis=1)


# In[5]:


customer_orders = customer.merge(orders_df, on='customer_id')


# Проведем эксплораторный анализ, для этого посмотрим тип данных датафрейма, его размер, проверим отсутствующие значения и количество уникальных значений.

# In[6]:


customer_orders.dtypes


# In[7]:


customer_orders.isna().sum()


# In[8]:


customer_orders.nunique()


# In[9]:


items = items_df.drop(['seller_id', 'freight_value'], axis=1)
items.head()


# Для начала, нам нужно рассчитать число пользователей, совершивших покупку только один раз. Мы будем учитывать тех пользователей, которым товар был доставлен. Покупкой товара считаем дату подтверждения оплаты заказа. Товар мог быть недоставлен или отменен, но это не отменяет факт платежа и соответственно покупки. За недоставленные/отменные заказы будут отвечать другие метрики.

# In[10]:


customer_orders .query('order_approved_at != "NaT"') .groupby('customer_unique_id', as_index=False) .agg({'customer_id': 'count'}) .sort_values('customer_id', ascending=False) .query('customer_id == 1').customer_id.sum()


# Ввиду того, что возникают случаи, при которых заказы не доставляются покупателям, стоит узнать среднее количество таких заказов.

# In[11]:


orders_cancaled = customer_orders.query('order_delivered_customer_date == "NaT" and order_status in ("unavailable", "canceled")')

orders_cancel = orders_cancaled.groupby(['order_estimated_delivery_date','order_status'], as_index=False) .agg({'order_id': 'count'}).fillna(0).sort_values('order_id')


# In[12]:


cancel = orders_cancel.groupby(['order_status', pd.Grouper(key="order_estimated_delivery_date", freq="M")]) .agg({'order_id': 'sum'}).reset_index()
cancel.head()


# In[13]:


avg = cancel.order_id.mean().round(1)
avg


# В среднем около 26 заказов в месяц не доставляются покупателям по причине отмены заказа и отстуствия товара. Параметрами для расчета послужили статусы: отменен и недоступен, а также отстутствие даты доставки покупателю. Все остальные статусы - это по сути маркеры, отражающие состояние заказа "в работе".
# Далее построим график, на котором будет отражена динамика недоставленных заказов относительно среднего. 

# In[14]:


plt.figure(figsize=(12, 6))
ax = sns.lineplot(x="order_estimated_delivery_date", y="order_id", hue="order_status", data=cancel)
ax.set(title='Динамика невыполненных доставок в разрезе причин', xlabel='период', 
       ylabel='кол-во поссылок, которые не были доставлены')
plt.legend(['Отменен','Недоступен'])

ax.axhline(y=26.127659574468087, color='r', linestyle='--', linewidth=2)


# Для того чтобы определить пиковую активность покупателей по дням по каждому товару, определим в какой день недели товар чаще всего покупается.

# In[15]:


good_d_sale = items.groupby('order_id', as_index=False).agg({'product_id': list}).explode('product_id')

good_wd_sale = good_d_sale.merge(orders_df, on='order_id') .drop(['order_estimated_delivery_date', 'order_delivered_customer_date', 'order_delivered_carrier_date',
      'order_approved_at', 'order_status', 'customer_id', 'order_id'], axis=1)


# In[16]:


good_wd_sale['order_purchase_timestamp'] = good_wd_sale['order_purchase_timestamp'].dt.day_name()
good_wd_sale.head()


# In[17]:


good_wd_sale.rename(columns={'order_purchase_timestamp': 'Day'}, inplace= True)
good_wd_sale.head()


# In[18]:


day_week = good_wd_sale.groupby('product_id').Day.value_counts().to_frame()


# In[19]:


day_week.rename(columns={'Day': 'Quantity'}, inplace= True)


# In[20]:


day_week.reset_index(inplace= True)


# In[21]:


best_day = day_week.groupby('product_id', as_index=False).agg({'Quantity': 'sum', 'Day': ', '.join}) .pivot(index='product_id', columns='Day', values='Quantity').fillna(0) .idxmax(axis=1).to_frame().reset_index()
best_day


# Посмотрим сколько покупок в среднем в неделю делает каждый пользователь (в разрезе по месяцам)

# In[22]:


orders = customer_orders .query('order_approved_at != "NaT"') .groupby(['customer_unique_id', pd.Grouper(key="order_estimated_delivery_date", freq="M")]) .agg({'customer_id': 'count'}) .sort_values('customer_id', ascending=False) .reset_index()

orders.head()


# In[23]:


orders['month'] = orders['order_estimated_delivery_date'].dt.month_name()
orders['year'] = orders['order_estimated_delivery_date'].dt.year
orders['week'] = orders['order_estimated_delivery_date'].dt.day / 7
orders.drop('order_estimated_delivery_date', axis=1)
orders.rename(columns={'customer_id': 'quantity_orders'}, inplace= True)

# Расчет среднего количества покупок в нелелю для каждого пользователя по месяцам:
orders['orders_in_week'] = orders['quantity_orders'] / orders['week'] 
orders.head()


# Проведем когортный анализ пользователей по дате первого заказа для каждого пользователя и попробуем выявить когорту с самым высоким retention на 3й месяц в период с января по декабрь.

# In[24]:


order_first = customer_orders.groupby('customer_unique_id', as_index=False) .agg({'order_approved_at': 'min'}).rename(columns={'order_approved_at': 'first_month'})


# In[25]:


order_first['first_month'] = order_first['first_month'].apply(lambda x: x.strftime('%Y-%m')if not pd.isnull(x) else '') 


# In[26]:


# Создаем колонку с месяцем заказа
customer_orders['order_month'] = customer_orders['order_approved_at'].apply(lambda x: x.strftime('%Y-%m')if not pd.isnull(x) else '') 


# In[27]:


# Мерджим основной датафрейм и информационный
customer_orders = customer_orders.merge(order_first, on='customer_unique_id')


# In[28]:


# Приводим данные по датам к единому формату
customer_orders['order_month'] = pd.to_datetime(customer_orders.order_month).dt.to_period('M')
customer_orders['first_month'] = pd.to_datetime(customer_orders.first_month).dt.to_period('M')


# In[29]:


customer_orders['month'] = customer_orders['order_month'].sub(customer_orders['first_month']).apply(lambda x: x.n if not pd.isnull(x) else '')


# In[30]:


orders_cohort = customer_orders.groupby(['customer_unique_id','order_month', 'first_month', 'month'], as_index=False) .agg({'order_id': 'count'})


# In[31]:


# Группируем по уникальным пользователям
grouped = orders_cohort.groupby(['first_month', 'order_month', 'month'])
cohorts = grouped.agg({'customer_unique_id': pd.Series.nunique, 'order_id': np.sum})
cohorts.rename(columns={'order_id': 'orders', 'customer_unique_id': 'users'}, inplace=True)
cohorts.head()


# In[32]:


# Добавляем порядковое значение сohort_period для каждого из order_period. 
# сohort_period равен порядковому номеру месяца в массиве начинающийся с 1
def cohort_period(orders_cohort):
    orders_cohort['cohort_period'] = np.arange(len(orders_cohort)) + 1
    return orders_cohort
cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts


# In[33]:


cohorts.reset_index(inplace=True)
cohorts.set_index(['first_month', 'cohort_period'], inplace=True)
cohorts


# In[34]:


# Создаем ряд содержаший размер каждой когорты start_month
cohort_size = cohorts['users'].groupby(level=0).first()
cohort_size


# In[35]:


# Рассчитываем долю возврата покупатлей от первоначального размера когорты.
cohort_users = cohorts['users'].unstack(1).divide(cohort_size, axis=0)


# In[36]:


ur_style = (cohort_users.style.set_caption('User retention by cohort') .background_gradient(cmap='viridis').highlight_null('white'))
ur_style.format("{:.2%}", na_rep="")


# Когорта от 2017-05 имеет самый высокий уровень retention на третий месяц

# Построим RFM-сегментацию пользователей, сформировав кластеры из следующих метрик: 
# R - время от последней покупки пользователя до текущей даты, 
# F - суммарное количество покупок у пользователя за всё время, 
# M - сумма покупок за всё время.

# In[37]:


# Поскольку временной период данных достаточно далек от настоящего времени, вместо текщей даты будем считать max+1
last_date = customer_orders['order_approved_at'].max() + timedelta(days=1)
last_date


# In[38]:


period = 365


# In[39]:


# Добавляем столбец с количеством дней между покупкой и настоящим моментом
customer_orders['DaysOrder'] = customer_orders['order_approved_at'].apply(lambda x: (last_date - x).days)


# In[40]:


# Строим RF
RF = customer_orders .groupby(['customer_unique_id', 'order_id']) .agg({'DaysOrder': lambda x: x.min(),
      'order_approved_at': lambda x: len([d for d in x if d >= last_date - timedelta(days=period)])})

RF.rename(columns={'DaysOrder': 'recency',
                    'order_approved_at': 'frequency'}, inplace=True)

RF.reset_index()


# In[41]:


payment = items .groupby(['order_id', 'price']) .agg({'order_id' : 'nunique'})

payment.rename(columns={'order_id' :'order_count'}, inplace=True)


# In[42]:


payment.reset_index(inplace=True)
payment.set_index(['order_id'], inplace=True)
payment.head()


# In[43]:


payment.dtypes


# In[44]:


# Рассчитыавем монетезацию продаж
payment['payment_price'] = payment.price * payment.order_count


# In[45]:


# Строим RFM-сегментацию пользователей
M = payment .groupby('order_id') .agg({'payment_price' : lambda x: x.sum()})

M.rename(columns={'payment_price' : 'monetary'}, inplace=True)

M.reset_index()


# In[46]:


RFM = RF.merge(M, on='order_id')


# In[47]:


RFM


# In[48]:


# Чтобы оценить каждого клиента, необходимо установить диапазоны для каждого кластера
quintiles = RFM[['recency', 'frequency', 'monetary']].quantile([.2, .4, .6, .8]).to_dict()
quintiles


# In[49]:


# Для каждого RFM-сегмента строим границы метрик для интерпретации кластеров
def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5  


# In[50]:


RFM['R'] = RFM['recency'].apply(lambda x: r_score(x))
RFM['F'] = RFM['frequency'].apply(lambda x: fm_score(x, 'frequency'))
RFM['M'] = RFM['monetary'].apply(lambda x: fm_score(x, 'monetary'))


# In[51]:


RFM['RFM_score'] = RFM['R'].map(str) + RFM['F'].map(str) + RFM['M'].map(str)
RFM.head()


# In[52]:


Segment = pd.Series(['Champions', 'Loyal Customers', 'Potential Loyalist', 'Recent Customers', 'Promising',
                         'Customers Needing Attention', 'About To Sleep', 'At Risk', 'Can’t Lose Them',
                           'Hibernating'])

Description = pd.Series(['Bought recently, buy often and spend the most',
                              'Buy on a regular basis. Responsive to promotions',
                              'Recent customers with average frequency',
                             'Bought most recently, but not often',
                             'Recent shoppers, but haven’t spent much',
                             'Above average recency, frequency and monetary values. May not have bought very recently though',
                             'Below average recency and frequency. Will lose them if not reactivated',
                             'Purchased often but a long time ago. Need to bring them back!',
                             'Used to purchase frequently but haven’t returned for a long time',
                             'Last purchase was long back and low number of orders. May be lost'])


# In[53]:


# Создаем словарь для таблицы и графика
dictionary = pd.DataFrame({'Segment': Segment, 'Description': Description })
dictionary


# In[54]:


segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

RFM['Segment'] = RFM['R'].map(str) + RFM['F'].map(str)
RFM['Segment'] = RFM['Segment'].replace(segt_map, regex=True)
RFM.head(10)


# In[55]:


# Считаем количество покупателей в каждом сегменте
segments_counts = RFM['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['champions', 'loyal customers']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )
plt.show()

