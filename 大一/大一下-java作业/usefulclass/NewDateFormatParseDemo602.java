package ch04.usefulclass;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

public class NewDateFormatParseDemo602 {

	public static void main(String[] args) {

		//// 创建LocalDateTime对象
		LocalDateTime dateTime = LocalDateTime.now();
		System.out.println("dateTime格式化之前：" + dateTime);

		// 设置格式化类
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
		String text = dateTime.format(formatter);
		System.out.println("dateTime格式化之后：" + text);

		// 格式化字符串"2018-08-18 08:58:18"，返回LocalDateTime对象
		LocalDateTime parsedDateTime = LocalDateTime.parse("2018-08-18 08:58:18", formatter);
		System.out.println("LocalDateTime解析之后：" + parsedDateTime);

		//// 创建LocalDate对象
		LocalDate date = LocalDate.now();
		System.out.println("date格式化之前：" + date);

		// 重新设置格式化类
		formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd");
		text = date.format(formatter);
		System.out.println("date格式化之后：" + text);

		// 格式化字符串"2018-08-18"，返回LocalDate对象
		LocalDate parsedDate = LocalDate.parse("2018-08-18", formatter);
		System.out.println("LocalDate解析之后：" + parsedDate);

		//// 创建LocalTime对象
		LocalTime time = LocalTime.now();
		System.out.println("time格式化之前：" + time);

		// 重新设置格式化类
		formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
		text = time.format(formatter);
		System.out.println("time格式化之后：" + text);

		// 格式化字符串"08:58:18"，返回LocalTime对象
		LocalTime parsedTime = LocalTime.parse("08:58:18", formatter);
		System.out.println("LocalTime解析之后：" + parsedTime);
	}
}
