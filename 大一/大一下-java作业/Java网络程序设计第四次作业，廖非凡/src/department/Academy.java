package department;

import java.util.Vector;

import Course.Course;
import main.CoRecord;
import main.Stage;
import person.Ruler;
import person.Student;

public class Academy extends Stage {
	Vector<College> departmentList = new Vector<College>();
	Vector<College> labList = new Vector<College>();
	Vector<College> instituteList = new Vector<College>();

	public Academy() {
	};

	public Academy(String name2) {
		name = name2;
	}

	public Academy(String name2, Ruler boss2, College firstT1, College firstT2, College firstT3) {
		name = name2;

		boss2.level = Ruler.L21;
		boss = boss2;

		firstT1.type = College.T1;
		departmentList.add(firstT1);
		firstT2.type = College.T2;
		labList.add(firstT2);
		firstT3.type = College.T1;
		instituteList.add(firstT3);

		ShowMe();
	}

	// 获得某个学生的各科分数和平均分
	public void GetScore(Student stu) {
		int average = 0;
		int num = 0;

		System.out.println("---" + stu.name + "期末报告单---");
		if (!IsOurAc(stu))
			System.out.println("(" + stu.name + "非本院学生，但由本院代理评分)");
		System.out.println();
		System.out.println();

		for (CoRecord s : stu.courseList) {
			if (s.score == -1) {
				System.out.println("	---" + s.course.name + ":" + "无有效成绩");
				continue;
			}
			System.out.println("	---" + s.course.name + ":" + s.score);
			average += s.score;
			num++;
		}
		if (num == 0) {
			System.out.println("无法求得平均值！");
			System.out.println("\n				" + name + "[盖章]\n\n\n");
			return;
		}
		average /= num;
		System.out.println("\n	---平均分:" + average);
		System.out.println("\n				" + name + "[盖章]\n\n\n");
	}

	// 获得某个科目的每个同学分数和平均分
	public void GetScore(Course course) {
		int average = 0;
		int num = 0;

		System.out.println("---" + course.name + "期末报告单---");
		System.out.println("科任老师：" + course.teacher);
		System.out.println();
		System.out.println();

		for (Student s : course.stuList) {
			for (CoRecord r : s.courseList) {
				if (r.course == course) {
					if (r.score == -1) {
						System.out.println("	---" + s.name + ":" + "无有效成绩");
						continue;
					}
					System.out.println("	---" + s.name + ":" + r.score);
					average += r.score;
					num++;
				}

			}
		}
		if (num == 0) {
			System.out.println("无法求得平均值！");
			System.out.println("\n				" + name + "[盖章]\n\n\n");
			return;
		}
		average /= num;
		System.out.println("\n	---平均分:" + average);
		System.out.println("\n				" + name + "[盖章]\n\n\n");
	}

	// 判断这个学生是不是本院的
	protected boolean IsOurAc(Student stu) {
		if (stu.academy == name)
			return true;
		return false;
	}

	@Override

	public void ShowMe() {
		System.out.println();
		System.out.println("\nAcademy:" + name + "\nDean:" + boss.name);
		System.out.println("Department:");
		if (!departmentList.isEmpty()) {
			for (College s : departmentList) {
				System.out.println("“" + s.name + "”");
			}
		}
		System.out.println("Lab:");
		if (!labList.isEmpty()) {
			for (College s : labList) {
				System.out.println("“" + s.name + "”");
			}
		}
		System.out.println("Institute:");
		if (!instituteList.isEmpty()) {
			for (College s : instituteList) {
				System.out.println("“" + s.name + "”");
			}
		}

	}

}
