/*
    Lite³: A JSON-Compatible Zero-Copy Serialization Format

    Copyright © 2025 Elias de Jong <elias@fastserial.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

      __ __________________        ____
    _  ___ ___/ /___(_)_/ /_______|_  /
     _  _____/ / __/ /_  __/  _ \_/_ < 
      ___ __/ /___/ / / /_ /  __/____/ 
           /_____/_/  \__/ \___/       
*/
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "lite3.h"


unsigned char buf[2048] __attribute__((aligned(LITE3_NODE_ALIGNMENT)));


int main()
{
	size_t buflen = 0;

	if (lite3_init_obj(buf, &buflen, sizeof(buf)) < 0) {
		perror("Failed to initialize object");
		return 1;
	}

	if (lite3_set_i64(buf, &buflen, 0, sizeof(buf), "user_id", 12345)				< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "username", "jdoe")				< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "email_address", "jdoe@example.com")		< 0
	|| lite3_set_bool(buf, &buflen, 0, sizeof(buf), "is_active", true)				< 0
	|| lite3_set_f64(buf, &buflen, 0, sizeof(buf), "account_balance", 259.75)			< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "signup_date_str", "2023-08-15")			< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "last_login_date_iso", "2025-09-13T13:20:00Z")	< 0
	|| lite3_set_i64(buf, &buflen, 0, sizeof(buf), "birth_year", 1996)				< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "phone_number", "+14155555671")			< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "preferred_language", "en")			< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "time_zone", "Europe/Berlin")			< 0
	|| lite3_set_i64(buf, &buflen, 0, sizeof(buf), "loyalty_points", 845)				< 0
	|| lite3_set_f64(buf, &buflen, 0, sizeof(buf), "avg_session_length_minutes", 14.3)		< 0
	|| lite3_set_bool(buf, &buflen, 0, sizeof(buf), "newsletter_subscribed", false)			< 0
	|| lite3_set_str(buf, &buflen, 0, sizeof(buf), "ip_address", "192.168.0.42")			< 0
	|| lite3_set_null(buf, &buflen, 0, sizeof(buf), "notes")					< 0) {
		perror("Failed to insert key");
		return 1;
	}

	int64_t user_id;
	lite3_str username;
	lite3_str email_address;
	bool is_active;
	double account_balance;
	lite3_str signup_date_str;
	lite3_str last_login_date_iso;
	int64_t birth_year;
	lite3_str phone_number;
	lite3_str preferred_language;
	lite3_str time_zone;
	int64_t loyalty_points;
	double avg_session_length_minutes;
	bool newsletter_subscribed;
	lite3_str ip_address;

	if (lite3_get_i64(buf, buflen, 0, "user_id", &user_id)						< 0
	|| lite3_get_str(buf, buflen, 0, "username", &username)						< 0
	|| lite3_get_str(buf, buflen, 0, "email_address", &email_address)				< 0
	|| lite3_get_bool(buf, buflen, 0, "is_active", &is_active)					< 0
	|| lite3_get_f64(buf, buflen, 0, "account_balance", &account_balance)				< 0
	|| lite3_get_str(buf, buflen, 0, "signup_date_str", &signup_date_str)				< 0
	|| lite3_get_str(buf, buflen, 0, "last_login_date_iso", &last_login_date_iso)			< 0
	|| lite3_get_i64(buf, buflen, 0, "birth_year", &birth_year)					< 0
	|| lite3_get_str(buf, buflen, 0, "phone_number", &phone_number)					< 0
	|| lite3_get_str(buf, buflen, 0, "preferred_language", &preferred_language)			< 0
	|| lite3_get_str(buf, buflen, 0, "time_zone", &time_zone)					< 0
	|| lite3_get_i64(buf, buflen, 0, "loyalty_points", &loyalty_points)				< 0
	|| lite3_get_f64(buf, buflen, 0, "avg_session_length_minutes", &avg_session_length_minutes)	< 0
	|| lite3_get_bool(buf, buflen, 0, "newsletter_subscribed", &newsletter_subscribed)		< 0
	|| lite3_get_str(buf, buflen, 0, "ip_address", &ip_address)					< 0) {
		perror("Failed to get key");
		return 1;
	}

	assert(user_id == 12345);
	assert(strcmp(LITE3_STR(buf, username), "jdoe") == 0);
	assert(strcmp(LITE3_STR(buf, email_address), "jdoe@example.com") == 0);
	assert(is_active == true);
	assert(account_balance == 259.75);
	assert(strcmp(LITE3_STR(buf, signup_date_str), "2023-08-15") == 0);
	assert(strcmp(LITE3_STR(buf, last_login_date_iso), "2025-09-13T13:20:00Z") == 0);
	assert(birth_year == 1996);
	assert(strcmp(LITE3_STR(buf, phone_number), "+14155555671") == 0);
	assert(strcmp(LITE3_STR(buf, preferred_language), "en") == 0);
	assert(strcmp(LITE3_STR(buf, time_zone), "Europe/Berlin") == 0);
	assert(loyalty_points == 845);
	assert(avg_session_length_minutes == 14.3);
	assert(newsletter_subscribed == false);
	assert(strcmp(LITE3_STR(buf, ip_address), "192.168.0.42") == 0);

	return 0;
}