package ch04.usefulclass;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

public class NewDateTimeDemo601 {

	public static void main(String[] args) {

		// 使用now方法获得LocalDate对象
		LocalDate date1 = LocalDate.now();
		System.out.println("date1 = " + date1);

		// 使用of方法获得LocalDate对象
		LocalDate date2 = LocalDate.of(2021, 5, 1);
		System.out.println("date2 = " + date2);

		// 使用now方法获得LocalTime对象
		LocalTime time1 = LocalTime.now();
		System.out.println("time1 = " + time1);

		// 使用of方法获得LocalTime对象
		LocalTime time2 = LocalTime.of(18, 30, 18);
		System.out.println("time2 = " + time2);

		// 使用now方法获得LocalDateTime对象
		LocalDateTime dateTime1 = LocalDateTime.now();
		System.out.println("dateTime1 = " + dateTime1);

		// 使用of方法获得LocalDateTime对象
		LocalDateTime dateTime2 = LocalDateTime.of(2021, 5, 1, 18, 30, 18);
		System.out.println("dateTime2 = " + dateTime2);
	}

}
